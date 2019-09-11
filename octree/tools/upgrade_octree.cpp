#include <cstring>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>

#include "cmd_flags.h"
#include "octree.h"
#include "filenames.h"

using std::vector;
using std::string;
using std::cout;
using cflags::Require;
const float kPI = 3.14159265f;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_bool(node_dis, kOptional, false, "Output per-node displacement");
DEFINE_bool(node_label, kOptional, false, "Whether has node label");
DEFINE_bool(split_label, kOptional, false, "Compute per node splitting label");
DEFINE_bool(adaptive, kOptional, false, "Build adaptive octree");
DEFINE_int(adp_depth, kOptional, 4, "The starting depth of adaptive octree");
DEFINE_bool(key2xyz, kOptional, true, "Convert the key to xyz when serialization");
DEFINE_bool(verbose, kOptional, true, "Output logs");


bool load_raw_data(vector<char>& data, const string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) return false;

  infile.seekg(0, infile.end);
  int len = infile.tellg();
  infile.seekg(0, infile.beg);

  data.resize(len);
  infile.read(data.data(), len);
  return true;
}

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: update_octree.exe");
    return 0;
  }
  if (FLAGS_node_label && FLAGS_split_label) {
    cout << "The node_label and split_label can not be true at the same time\n";
    return 0;
  }

  // file path
  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  else output_path = extract_path(file_path);
  output_path += "/";

  vector<string> all_files;
  get_all_filenames(all_files, file_path);

  for (int i = 0; i < all_files.size(); i++) {
    vector<char> oct_input;
    bool succ = load_raw_data(oct_input, all_files[i]);
    if (!succ) continue;

    /// parse the octree
    int* octi = (int*)oct_input.data();
    int total_node_num = octi[0];
    int final_node_num = octi[1];
    int depth = octi[2];
    int full_layer = octi[3];
    const int* node_num = octi + 4;
    const int* node_num_cum = node_num + depth + 1;
    const int* key = node_num_cum + depth + 2;
    const int* children = key + total_node_num;
    const int* data = children + total_node_num;
    const float* normal_ptr = (const float*)data;

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
    float bbmin[] = {0.0f, 0.0f, 0.0f };
    float bbmax[] = {width, width, width };
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

    octree_info.set_nnum(node_num);
    octree_info.set_nnum_cum();
    octree_info.set_nempty(nnum_nempty.data());
    octree_info.set_ptr_dis();

    /// output octree
    vector<char> buffer(octree_info.sizeof_octree());
    memcpy(buffer.data(), &octree_info, sizeof(OctreeInfo));
    Octree oct_parser;
    oct_parser.set_octree(buffer);
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
    string filename = extract_filename(all_files[i]);
    oct_parser.write_octree(output_path + filename + ".upgrade.octree");
  }

  return 0;
}

