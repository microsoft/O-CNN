#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <limits>
#include <ctime>

#include "math_functions.h"
#include "filenames.h"
#include "points.h"
#include "octree.h"
#include "contour.h"
#include "cmd_flags.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_string(axis, kOptional, "y", "The upright axis of the input model");
DEFINE_bool(verbose, kOptional, true, "Output logs");

class RandAxis {
 public:
  RandAxis(float upright[3])
    : generator_(static_cast<unsigned int>(time(nullptr))),
      distribution_(-1.0, 1.0) {
    for (int i = 0; i < 3; ++i) { upright_dir_[i] = upright[i]; }
  }

  void operator()(float* axis) {
    float dot = 1.0f;
    while (dot > 0) {
      for (int i = 0; i < 3; ++i) { axis[i] = distribution_(generator_); }
      float len = norm2(axis, 3);
      if (len < 1.0e-6) continue; // ignore zero vector
      for (int i = 0; i < 3; ++i) { axis[i] /= len; }
      dot = dot_prod(axis, upright_dir_);
    }
  }

 private:
  float upright_dir_[3];
  std::default_random_engine generator_;
  std::uniform_real_distribution<float> distribution_;
};


class ZBuffer {
 public:
  ZBuffer() : scale_(1) {}

  void scan(Octree& octree_out, const Octree& octree_in, const vector<float> axis) {
    // drop_flags: 1 - drop; 0 - keep, iff the node is visible and non-empty
    int depth = octree_in.info().depth();
    int num = octree_in.info().node_num(depth);
    vector<vector<int> > drop_flags(depth + 1);
    drop_flags[depth].resize(num, 1);
    
    const int axis_channel = 3;
    int axis_num = axis.size() / axis_channel;
    for (int i = 0; i < axis_num; ++i) {
      // set the scanning coordinate system
      // if rot_channel == 3, then it is an axis, i.e. only z axis
      // if rot_channel == 9, then it is an rotation matrix
      set_axis(axis.data() + i * axis_channel, axis_channel);

      // calc the bound of the x-y projection plane
      bbox_xy(depth);

      // run the z_buffer algorithm
      vector<int> flags(num, 1);
      z_buffer(flags, octree_in);
      for (int j = 0; j < num; ++j) {
        if (flags[j] == 0) drop_flags[depth][j] = 0;
      }
    }

    // generate drop flags for other octree layers
    generate_flags(drop_flags, octree_in);

    // drop tree according to the flags
    trim_octree(octree_out, octree_in, drop_flags);
  }

  void set_scale(float scale) {
    scale_ = scale;
  }

  void set_axis(const float* axis, int n = 3) {
    if (n == 3) {
      for (int i = 0; i < 3; ++i) { z_[i] = axis[i]; }
      axes(x_, y_, z_);
    } else { // n == 9
      x_[0] = axis[0];    x_[1] = axis[1];    x_[2] = axis[2];
      y_[0] = axis[3];    y_[1] = axis[4];    y_[2] = axis[5];
      z_[0] = axis[6];    z_[1] = axis[7];    z_[2] = axis[8];
    }
  }

 protected:

  void bbox_xy(int depth) {
    for (int i = 0; i < 3; ++i) {
      bbmin_[i] = std::numeric_limits<float>::max();
      bbmax_[i] = -std::numeric_limits<float>::max();
    }

    for (int i = 0; i < 8; ++i) {
      float pt[3] = { i / 4, (i / 2) % 2, i % 2 };
      float x = dot_prod(pt, x_);
      if (x < bbmin_[0]) bbmin_[0] = x;
      if (x > bbmax_[0]) bbmax_[0] = x;

      float y = dot_prod(pt, y_);
      if (y < bbmin_[1]) bbmin_[1] = y;
      if (y > bbmax_[1]) bbmax_[1] = y;

      float z = dot_prod(pt, z_);
      if (z < bbmin_[2]) bbmin_[2] = z;
      if (z > bbmax_[2])bbmax_[2] = z;
    }

    float width = -1;
    float scale = 1 << depth;
    for (int i = 0; i < 3; ++i) {
      bbmax_[i] *= scale;
      bbmin_[i] *= scale;

      float dis = bbmax_[i] - bbmin_[i];
      if (dis > width) width = dis;
    }

    width_ = static_cast<int>(width * scale_) + 2; // slightly larger
  }

  void reset_buffer() {
    id_buffer_.assign(width_ * width_, -1);
    z_buffer_.assign(width_ * width_, std::numeric_limits<float>::max());
  }

  void z_buffer(vector<int>& drop_flags, const Octree& octree_in) {
    // reset the buffer
    reset_buffer();

    const int depth = octree_in.info().depth();
    const int num = octree_in.info().node_num(depth);

    const unsigned int* key = octree_in.key_cpu(depth);
    const int* children = octree_in.children_cpu(depth);
    for (int i = 0; i < num; ++i) {
      if (children[i] < 0) continue; // empty node

      float normal[3];
      octree_in.node_normal(normal, i, depth);
      if (dot_prod(normal, z_) >= 0) continue; // invisible node

      float pt[3];
      octree_in.key2xyz(pt, key[i], depth);
      float x = (dot_prod(x_, pt) - bbmin_[0]) * scale_;
      float y = (dot_prod(y_, pt) - bbmin_[1]) * scale_;
      float z = dot_prod(z_, pt);

      int k = static_cast<int>(x + 0.5f) * width_ + static_cast<int>(y + 0.5f);
      if (z_buffer_[k] > z) {
        z_buffer_[k] = z;
        int id = id_buffer_[k];
        if (id >= 0) {
          drop_flags[id] = 1;
        }
        id_buffer_[k] = i;
        drop_flags[i] = 0;
      }
    }
  }

  void generate_flags(vector<vector<int>>& drop_flags, const Octree& octree_in) {
    int depth = octree_in.info().depth();
    int depth_full = octree_in.info().full_layer();
    for (int d = 0; d < depth_full; ++d) {
      // keep all the nodes whose depth are smaller than depth_full
      drop_flags[d].assign(1 << (3 * d), 0);
    }
    for (int d = depth - 1; d >= depth_full; --d) {
      int num = octree_in.info().node_num(d);
      drop_flags[d].assign(num, 1);
      const int* child_d = octree_in.children_cpu(d);
      for (int i = 0; i < num; ++i) {
        int j = child_d[i];
        if (j < 0) continue; // empty node

        int drop = 1;
        for (int k = j * 8; k < j * 8 + 8; ++k) {
          // keep the node if any of its children nodes is kept
          if (drop_flags[d + 1][k] == 0) {
            drop = 0;
            break;
          }
        }
        drop_flags[d][i] = drop;
      }
    }
  }

  void trim_octree(Octree& octree_out, const Octree& octree_in,
      vector<vector<int>>& drop_flags) {
    // calculate the node number for the octree_out
    int depth = octree_in.info().depth();
    int depth_full = octree_in.info().full_layer();
    vector<int> node_num_nempty(depth + 1, 0);
    for (int d = 0; d < depth_full; ++d) {
      node_num_nempty[d] = 1 << (3 * d);
    }
    for (int d = depth_full; d <= depth; ++d) {
      int num = 0;
      for (auto v : drop_flags[d]) {
        if (v == 0) num++;
      }
      node_num_nempty[d] = num;
    }
    vector<int> node_num(depth + 1, 0);
    node_num[0] = 1;
    for (int d = 1; d <= depth; ++d) {
      node_num[d] = 8 * node_num_nempty[d - 1];
    }

    // initialize
    OctreeInfo info_out = octree_in.info();
    info_out.set_nnum(node_num.data());
    info_out.set_nempty(node_num_nempty.data());
    info_out.set_nnum_cum();
    info_out.set_ptr_dis();
    octree_out.resize_octree(info_out.sizeof_octree());
    octree_out.mutable_info() = info_out;

    // copy data
    // !!! current channel_key = 1
    int channel_feature = octree_in.info().channel(OctreeInfo::kFeature);
    int location_feature = octree_in.info().locations(OctreeInfo::kFeature);
    int channel_label = octree_in.info().channel(OctreeInfo::kLabel);
    int location_label = octree_in.info().locations(OctreeInfo::kLabel);
    int channel_split = octree_in.info().channel(OctreeInfo::kSplit);
    int location_split = octree_in.info().locations(OctreeInfo::kSplit);

    for (int d = 1; d <= depth; ++d) {
      int num = octree_in.info().node_num(d - 1);
      vector<int>& drop = drop_flags[d - 1];
      vector<int>& drop_d = drop_flags[d];

      // copy children and key
      // !!! Caveat: currently channel_key is 1, and channel_child is 1
      const int* child_in = octree_in.children_cpu(d - 1);
      const unsigned int* key_in = octree_in.key_cpu(d);
      int* child_out = octree_out.mutable_children_cpu(d);
      unsigned int* key_out = octree_out.mutable_key_cpu(d);
      for (int i = 0, j = 0, id = 0; i < num; ++i) {
        // the node is dropped or empty
        if (drop[i] == 1) continue;
        int t = child_in[i];

        for (int k = 8 * t; k < 8 * t + 8; ++k) {
          key_out[j] = key_in[k];

          // the node is non-empty and kept
          int ch = drop_d[k] == 0 ? id++ : -1;
          child_out[j] = ch;

          j++;
        }
      }

      // copy feature
      if (location_feature == -1 || d == depth) {
        int nnum_in = octree_in.info().node_num(d);
        int nnum_out = octree_out.info().node_num(d);
        const float * feature_in = octree_in.feature_cpu(d);
        float* feature_out = octree_out.mutable_feature_cpu(d);
        for (int i = 0, j = 0; i < num; ++i) {
          // the node is dropped or empty
          if (drop[i] == 1) continue;
          int t = child_in[i];

          for (int k = 8 * t; k < 8 * t + 8; ++k) {
            for (int c = 0; c < channel_feature; ++c) {
              feature_out[c * nnum_out + j] = drop_d[k] == 0 ?
                  feature_in[c * nnum_in + k] : 0;
            }
            j++;
          }
        }
      }

      // copy label
      if ((location_label == -1 || d == depth) && channel_label != 0) {
        const float * label_in = octree_in.label_cpu(d);
        float* label_out = octree_out.mutable_label_cpu(d);
        for (int i = 0, j = 0; i < num; ++i) {
          // the node is dropped or empty
          if (drop[i] == 1) continue;
          int t = child_in[i];

          for (int k = 8 * t; k < 8 * t + 8; ++k) {
            label_out[j] = drop_d[k] == 0 ? label_in[k] : -1;
            ++j;
          }
        }
      }

      // copy split
      if ((location_split == -1 || d == depth) && channel_split != 0) {
        const float * split_in = octree_in.split_cpu(d);
        float* split_out = octree_out.mutable_split_cpu(d);
        for (int i = 0, j = 0; i < num; ++i) {
          // the node is dropped or empty
          if (drop[i] == 1) continue;
          int t = child_in[i];

          for (int k = 8 * t; k < 8 * t + 8; ++k) {
            split_out[j] = drop_d[k] == 0 ? split_in[k] : 0;
            ++j;
          }
        }
      }
    }
  }

 protected:
  float scale_;   // scale_ must be large than 0
  int width_;
  vector<int> id_buffer_;
  vector<float> z_buffer_;

  float bbmin_[3], bbmax_[3];
  float x_[3], y_[3], z_[3];
};


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

  float upright[] = { 0.0f, 0.0f, 0.0f };
  if (FLAGS_axis == "x") upright[0] = 1.0f;
  else if (FLAGS_axis == "y") upright[1] = 1.0f;
  else upright[2] = 1.0f;
  RandAxis rand_axis(upright);

  vector<string> all_files;
  get_all_filenames(all_files, file_path);
  for (int i = 0; i < all_files.size(); i++) {
    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;

    // load octree
    Octree octree;
    bool succ = octree.read_octree(all_files[i]);
    if (!succ) {
      cout << "Can not load " << filename << std::endl;
      continue;
    }
    string msg;
    succ = octree.info().check_format(msg);
    if (!succ) {
      cout << filename << ": format error!\n" << msg << std::endl;
      continue;
    }

    // dropout
    ZBuffer zbuffer;
    Octree octree_out;
    vector<float> axis = { 0, 0, 1, 0, 0, 1 };
    rand_axis(axis.data());
    rand_axis(axis.data() + 3);
    zbuffer.scan(octree_out, octree, axis);

    // save octree
    string filename_output = output_path + filename + ".zbuffer.octree";
    octree_out.write_octree(filename_output);
  }

  return 0;
}
