#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <limits>
#include <ctime>

#include "filenames.h"
#include "points.h"
#include "octree.h"
#include "contour.h"
#include "cmd_flags.h"
#include "types.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_string(axis, kOptional, "y", "The upright axis of the input model");
DEFINE_bool(verbose, kOptional, true, "Output logs");

// !!! todo: merge the code with octree zbuffer

void prune_octree(Octree& octree_out, const Octree& octree_in) {
  const int bnd0[7][3] = { { 1, 1, 1 },
    { 2, 2, 2 },    { 4, 4, 4 },    { 8, 6, 8 },
    { 16, 10, 16 }, { 30, 18, 30 }, { 60, 36, 60 }
  };
  const int bnd1[7][3] = { { 1, 1, 1 },
    { 2, 2, 2 },   { 4, 3, 4 },    { 8, 5, 8 },
    { 15, 9, 15 }, { 30, 18, 30 }, { 60, 36, 60 }
  };

  // 0, 1, 2, 3, 4, 5, 6
  vector<int> nnum_prune = { 1, 8, 64, 384, 2560, 16200, 129600 };
  vector<int> nnum_nempty_prune = { 1, 8, 48, 320, 2025, 16200, 129600 };

  // for the full layer
  int depth = octree_in.info().depth();
  int depth_full = octree_in.info().full_layer();
  // CHECK(depth_full > 1);

  // calculate the node number for the octree_out
  vector<int> nnum(depth + 1, 0), nnum_nempty(depth + 1, 0);
  for (int d = 0; d <= depth; ++d) {
    nnum[d] = d > depth_full ? octree_in.info().node_num(d) : nnum_prune[d];
    nnum_nempty[d] = d >= depth_full ? octree_in.info().node_num_nempty(d) : nnum_nempty_prune[d];
  }

  // init
  OctreeInfo info_out = octree_in.info();
  info_out.set_nnum(nnum.data());
  info_out.set_nempty(nnum_nempty.data());
  info_out.set_nnum_cum();
  info_out.set_ptr_dis();
  info_out.set_full_layer(2); // !!! full layer
  octree_out.resize_octree(info_out.sizeof_octree());
  octree_out.mutable_info() = info_out;

  int channel_feature = octree_in.info().channel(OctreeInfo::kFeature);
  int location_feature = octree_in.info().locations(OctreeInfo::kFeature);
  int channel_label = octree_in.info().channel(OctreeInfo::kLabel);
  int location_label = octree_in.info().locations(OctreeInfo::kLabel);
  int channel_split = octree_in.info().channel(OctreeInfo::kSplit);
  int location_split = octree_in.info().locations(OctreeInfo::kSplit);

  for (int d = 1; d <= depth_full; ++d) {
    int id = 0, j = 0;
    int* child_out = octree_out.mutable_children_cpu(d);
    const int* child_in = octree_in.children_cpu(d);
    uintk* key_out = octree_out.mutable_key_cpu(d);
    float* split_out = octree_out.mutable_split_cpu(d);

    for (int i = 0; i < octree_in.info().node_num(d); ++i) {
      uintk key = i, pt[3];
      octree_in.key2xyz(pt, key, d);
      if (pt[0] < bnd0[d][0] && pt[1] < bnd0[d][1] &&
          pt[2] < bnd0[d][2]) {
        if (pt[0] < bnd1[d][0] && pt[1] < bnd1[d][1] &&
            pt[2] < bnd1[d][2]) {
          child_out[j] = child_in[i] < 0 ? -1 : id++;
        } else {
          child_out[j] = -1; // empty
        }

        key_out[j] = key;

        if ((location_split == -1 || d == depth) && channel_split != 0) {
          split_out[j] = child_out[j] < 0 ? 0.0f : 1.0f;
        }

        // update j
        j++;
      }
    }
  }


  // copy directly
  for (int d = depth; d > depth_full; --d) {
    int num = octree_in.info().node_num(d);

    const int* child_in = octree_in.children_cpu(d);
    int* child_out = octree_out.mutable_children_cpu(d);
    std::copy_n(child_in, num, child_out);

    const uintk* key_in = octree_in.key_cpu(d);
    uintk* key_out = octree_out.mutable_key_cpu(d);
    std::copy_n(key_in, num, key_out);

    if (location_feature == -1 || d == depth) {
      const float * feature_in = octree_in.feature_cpu(d);
      float* feature_out = octree_out.mutable_feature_cpu(d);
      std::copy_n(feature_in, num * channel_feature, feature_out);
    }

    if ((location_label == -1 || d == depth) && channel_label != 0) {
      const float * label_in = octree_in.label_cpu(d);
      float* label_out = octree_out.mutable_label_cpu(d);
      std::copy_n(label_in, num, label_out);
    }

    if ((location_split == -1 || d == depth) && channel_split != 0) {
      const float * split_in = octree_in.split_cpu(d);
      float* split_out = octree_out.mutable_split_cpu(d);
      std::copy_n(split_in, num, split_out);
    }
  }
}


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: octree2mesh.exe");
    return 0;
  }

  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  else output_path = extract_path(file_path);
  output_path += "/";

  vector<string> all_files;
  get_all_filenames(all_files, file_path);
  for (int i = 0; i < all_files.size(); i++) {
    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;

    // load octree
    Octree octree_in;
    bool succ = octree_in.read_octree(all_files[i]);
    if (!succ) {
      cout << "Can not load " << filename << std::endl;
      continue;
    }
    string msg;
    succ = octree_in.info().check_format(msg);
    if (!succ) {
      cout << filename << ": format error!\n" << msg << std::endl;
      continue;
    }

    // prune full layer
    Octree octree_out;
    prune_octree(octree_out, octree_in);

    // save octree
    string filename_output = output_path + filename + ".prune.octree";
    octree_out.write_octree(filename_output);
  }

  return 0;
}
