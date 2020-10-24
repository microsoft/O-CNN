#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/octree.hpp"

using namespace caffe;
using std::vector;
using std::ofstream;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb", "The backend {leveldb, lmdb} of the databases");
DEFINE_int32(data_num, -1, "The number of data that is dumped to the disk");

void revert_octree_database(const string& db_name_input, string& path_output) {
  scoped_ptr<db::DB> db_input(db::GetDB(FLAGS_backend));
  db_input->Open(db_name_input, db::READ);
  scoped_ptr<db::Cursor> cursor_input(db_input->NewCursor());

  LOG(INFO) << "Starting ...";
  int count = 0;
  Datum datum_input;
  string datalist;
  while (cursor_input->valid()) {
    datum_input.ParseFromString(cursor_input->value());
    const string& octree_input = datum_input.data();
    string key_str = cursor_input->key();

    // update datalist
    char buffer[64];
    sprintf(buffer, " %d\n", datum_input.label());
    datalist += key_str + buffer;

    // dump the octree to disk
    if (FLAGS_data_num == -1 || count < FLAGS_data_num) {
      string filename = key_str;
      for (int i = 0; i < filename.size(); ++i) {
        if (filename[i] == '\\' || filename[i] == '/') filename[i] = '_';
      }
      ofstream outfile(path_output + filename, ofstream::binary);
      CHECK(outfile) << "Fail to open file " << filename;
      outfile.write(octree_input.data(), octree_input.size());
      outfile.close();
    }

    // next datum
    cursor_input->Next();
    if (++count % 1000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  if (count % 10000 != 0)	{
    LOG(INFO) << "Processed " << count << " files.";
  }

  ofstream outfile(path_output + "datalist.txt");
  CHECK(outfile) << path_output + "datalist.txt";
  outfile.write(datalist.data(), datalist.size());
  outfile.close();
}

int main(int argc, char** argv) {

  ::google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;

  gflags::SetUsageMessage("\nUsage:\n"
      "\trevert_octree_database [FLAGS] INPUT_DB OUTPUT_PATH/\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3 || argc > 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/revert_octree_database");
    return 1;
  }

  string db_input(argv[1]), path_output(argv[2]);
  revert_octree_database(db_input, path_output);

  return 0;
}
