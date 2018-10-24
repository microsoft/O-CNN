//
// This script converts the ModelNet dataset to the leveldb/lmdb format used
// by caffe to perform classification.


#include <fstream>
#include <string>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/rng.hpp"					// shuffle

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
using std::pair;
namespace db = caffe::db;

DEFINE_bool(shuffle, true,
    "Randomly shuffle the order of images and their labels");
DEFINE_bool(remove, false,
    "Remove the original octree when generating lmdb");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");

void ReadOctreeToDatum(const string& filename, const int label, Datum* datum) {
  size_t size;
  std::ifstream file(filename.c_str(),
      std::ios::in | std::ios::binary | std::ios::ate);
  CHECK(file) << "Unable to open train file #" << filename;
  size = file.tellg();
  std::string buffer(size, ' ');
  file.seekg(0, std::ios::beg);
  file.read(&buffer[0], size);
  file.close();
  datum->set_label(label);
  datum->set_data(buffer);
  datum->set_channels(size);
  datum->set_height(1);
  datum->set_width(1);
  // std::cout << "Size:\t" << size << std::endl;
}

void convert_dataset(const string& root_folder, const string& list_file,
    const string& db_name) {

  std::ifstream infile(list_file);
  std::vector<std::pair<std::string, int> > lines;
  std::string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(std::make_pair(line.substr(0, pos), label));
  }
  infile.close();

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    caffe::shuffle(lines.begin(), lines.end());

    // output shuffled data
    string list_file_shuffle = list_file;
    size_t p = list_file.rfind('.');
    if (string::npos == p) p = list_file.length();
    list_file_shuffle.insert(p, "_shuffle");
    std::ofstream outfile(list_file_shuffle);
    for (int line_id = 0; line_id < lines.size(); ++line_id) {
      outfile << lines[line_id].first << " "
          << lines[line_id].second << std::endl;
    }
    outfile.close();
  }
  LOG(INFO) << "A total of " << lines.size() << " octree files.";

  // Create new DB
  LOG(INFO) << "Writing data to DB";
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(db_name, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Data buffer
  Datum datum;
  int count = 0;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    string filename = root_folder + lines[line_id].first;
    ReadOctreeToDatum(filename, lines[line_id].second, &datum);

    // delete the file to save disk space
    if (FLAGS_remove) remove(filename.c_str());

    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
}

// convert the database from one to another
void convert_dataset(const string& db_name_src, const string& db_name_des) {
  // Create new DB
  LOG(INFO) << "Writing data to DB";
  scoped_ptr<db::DB> db_des(db::GetDB(FLAGS_backend));
  db_des->Open(db_name_des, db::NEW);
  scoped_ptr<db::Transaction> txn(db_des->NewTransaction());

  LOG(INFO) << "Load data to DB";
  scoped_ptr<db::DB> db_src(db::GetDB(FLAGS_backend));
  db_src->Open(db_name_src, db::READ);
  scoped_ptr<db::Cursor> cursor_(db_src->NewCursor());

  // Data buffer
  Datum datum;
  int count = 0;
  while (cursor_->valid()) {
    datum.ParseFromString(cursor_->value());
    
    /////////////////////////////
    // do something to the datum
    //////////////////////////////

    string out;
    CHECK(datum.SerializeToString(&out));
    string key_str = cursor_->key();
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db_des->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }

    //LOG(INFO) << key_str << " " << label;
    cursor_->Next();
  }

  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
}

int main(int argc, char** argv) {

  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

  gflags::SetUsageMessage("This script converts the ModelNet dataset to\n"
      "the leveldb/lmdb format used by caffe to perform classification.\n"
      "Usage:\n"
      "    convert_octree [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "convert_modelnet_data");
    return 1;
  }

  convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));

  return 0;
}