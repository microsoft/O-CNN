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
using std::copy;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb", "The backend {leveldb, lmdb} of the databases");
DEFINE_bool(node_dis, false, "Output per-node displacement");
DEFINE_bool(node_label, false, "Whether to have node label");
DEFINE_bool(split_label, false, "Compute per node splitting label");
DEFINE_bool(adaptive, false, "Build adaptive octree");
DEFINE_bool(key2xyz, true, "Convert the key to xyz when serialization");
DEFINE_int32(adp_depth, 4, "The starting depth of adaptive octree");


void upgrade_octree(string& octree_output, const string& octree_input) {
  /// parse the octree
  const int* octi = reinterpret_cast<const int*>(octree_input.data());
  int total_node_num = octi[0];
  int final_node_num = octi[1];
  int depth = octi[2];
  int full_layer = octi[3];
  const int* node_num = octi + 4;
  const int* node_num_cum = node_num + depth + 1;
  const int* key = node_num_cum + depth + 2;
  const int* children = key + total_node_num;
  const int* data = children + total_node_num;

  vector<int> nnum_nempty(depth + 1, 0);
  for (int d = 0; d <= depth; ++d) {
    // find the last element which is not equal to -1
    const int* children_d = children + node_num_cum[d];
    for (int i = node_num[d] - 1; i >= 0; i--) {
      if (children_d[i] != -1) {
        nnum_nempty[d] = children_d[i] + 1;
        break;
      }
    }
  }

  /// set octree info
  OctreeInfo octree_info;
  octree_info.set_batch_size(1);
  octree_info.set_depth(depth);
  octree_info.set_full_layer(full_layer);
  octree_info.set_adaptive_layer(FLAGS_adp_depth);
  octree_info.set_adaptive(FLAGS_adaptive);
  octree_info.set_node_dis(FLAGS_node_dis);
  octree_info.set_key2xyz(FLAGS_key2xyz);
  octree_info.set_threshold_normal(0.0f);
  octree_info.set_threshold_dist(0.0f);

  float width = static_cast<float>(1 << depth);
  float bbmin[] = { 0.0f, 0.0f, 0.0f };
  float bbmax[] = { width, width, width };
  octree_info.set_bbox(bbmin, bbmax);

  // by default, the octree contains Key and Child
  int channel = 1;
  octree_info.set_property(OctreeInfo::kKey, channel, -1);
  octree_info.set_property(OctreeInfo::kChild, channel, -1);

  // set feature
  int data_channel = FLAGS_node_dis ? 4 : 3;
  int location = FLAGS_adaptive ? -1 : depth;
  octree_info.set_property(OctreeInfo::kFeature, data_channel, location);

  // set label
  if (FLAGS_node_label) {
    octree_info.set_property(OctreeInfo::kLabel, 1, depth);
  }

  // set split label
  octree_info.set_property(OctreeInfo::kSplit, 0, 0);
  if (FLAGS_split_label) {
    octree_info.set_property(OctreeInfo::kSplit, 1, -1);
  }

  // update node_num
  octree_info.set_nnum(node_num);
  octree_info.set_nnum_cum();
  octree_info.set_nempty(nnum_nempty.data());
  octree_info.set_ptr_dis();

  /// output octree
  octree_output.resize(octree_info.sizeof_octree());
  OctreeParser oct_parser;
  oct_parser.set_cpu(&octree_output[0], &octree_info);
  int total_nnum = octree_info.total_nnum();
  int nnum_depth = octree_info.node_num(depth);
  memcpy(oct_parser.mutable_key_cpu(0), key, total_nnum * sizeof(int));
  memcpy(oct_parser.mutable_children_cpu(0), children, total_nnum * sizeof(int));
  int data_size = FLAGS_adaptive ? total_nnum : nnum_depth;
  memcpy(oct_parser.mutable_feature_cpu(0), data, data_size * data_channel * sizeof(float));
  if (FLAGS_node_label) {
    const int* ptr = data + data_size * data_channel;
    float* des = oct_parser.mutable_label_cpu(0);
    for (int i = 0; i < nnum_depth; ++i) {
      des[i] = static_cast<float>(ptr[i]);
    }
  }
  if (FLAGS_split_label) {
    const int* ptr = data + data_size * data_channel;
    float* des = oct_parser.mutable_split_cpu(0);
    for (int i = 0; i < total_nnum; ++i) {
      des[i] = static_cast<float>(ptr[i]);
    }
  }
}

void upgrade_octree_database(const string& db_name_input, const string& db_name_output) {
  scoped_ptr<db::DB> db_input(db::GetDB(FLAGS_backend));
  db_input->Open(db_name_input, db::READ);
  scoped_ptr<db::Cursor> cursor_input(db_input->NewCursor());

  LOG(INFO) << "Writing data to DB";
  scoped_ptr<db::DB> db_output(db::GetDB(FLAGS_backend));
  db_output->Open(db_name_output, db::NEW);
  scoped_ptr<db::Transaction> txn_output(db_output->NewTransaction());

  LOG(INFO) << "Starting ...";
  int count = 0;
  string octree_output, out_str;
  Datum datum_input, datum_output;
  while (cursor_input->valid()) {
    string key_str = cursor_input->key();
    datum_input.ParseFromString(cursor_input->value());
    const string& octree_input = datum_input.data();

    upgrade_octree(octree_output, octree_input);

    datum_output.set_label(datum_input.label());
    datum_output.set_data(octree_output);
    datum_output.set_channels(octree_output.size());
    datum_output.set_height(1);
    datum_output.set_width(1);

    CHECK(datum_output.SerializeToString(&out_str));
    txn_output->Put(key_str, out_str);

    cursor_input->Next();
    if (++count % 1000 == 0) {
      txn_output->Commit();
      txn_output.reset(db_output->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  if (count % 10000 != 0) {
    txn_output->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

  gflags::SetUsageMessage("\nUsage:\n"
      "\tupgrade_octree_database [FLAGS] INPUT_DB OUTPUT_DB\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3 || argc > 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/upgrade_octree_database");
    return 1;
  }

  CHECK(!(FLAGS_node_label && FLAGS_split_label))
      << "The node_label and split_label can not be true at the same time\n";

  string db_input(argv[1]), db_output(argv[2]);
  upgrade_octree_database(db_input, db_output);

  return 0;
}
