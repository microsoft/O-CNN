#include "transform_octree.h"

#include <algorithm>
#include <ctime>
#include <limits>
#include <random>

#include "math_functions.h"
#include "types.h"

void ScanOctree::set_scale(float scale) { scale_ = scale; }

void ScanOctree::set_axis(const float* axis, int n) {
  if (n == 3) {
    for (int i = 0; i < 3; ++i) { z_[i] = axis[i]; }
    axes(x_, y_, z_);
  } else { // n == 9
    x_[0] = axis[0];    x_[1] = axis[1];    x_[2] = axis[2];
    y_[0] = axis[3];    y_[1] = axis[4];    y_[2] = axis[5];
    z_[0] = axis[6];    z_[1] = axis[7];    z_[2] = axis[8];
  }
}

void ScanOctree::scan(vector<char>& octree_out, const OctreeParser& octree_in,
    const vector<float>& axis) {
  // drop_flags: 1 - drop; 0 - keep, iff the node is visible and non-empty
  int depth = octree_in.info().depth();
  int num = octree_in.info().node_num(depth);
  vector<vector<int>> drop_flags(depth + 1);
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

void ScanOctree::bbox_xy(int depth) {
  for (int i = 0; i < 3; ++i) {
    bbmin_[i] = std::numeric_limits<float>::max();
    bbmax_[i] = -std::numeric_limits<float>::max();
  }

  for (int i = 0; i < 8; ++i) {
    float pt[3] = {float(i / 4), float((i / 2) % 2), float(i % 2)};
    float x = dot_prod(pt, x_);
    if (x < bbmin_[0]) bbmin_[0] = x;
    if (x > bbmax_[0]) bbmax_[0] = x;

    float y = dot_prod(pt, y_);
    if (y < bbmin_[1]) bbmin_[1] = y;
    if (y > bbmax_[1]) bbmax_[1] = y;

    float z = dot_prod(pt, z_);
    if (z < bbmin_[2]) bbmin_[2] = z;
    if (z > bbmax_[2]) bbmax_[2] = z;
  }

  float width = -1;
  float scale = 1 << depth;
  for (int i = 0; i < 3; ++i) {
    bbmax_[i] *= scale;
    bbmin_[i] *= scale;

    float dis = bbmax_[i] - bbmin_[i];
    if (dis > width) width = dis;
  }

  width_ = static_cast<int>(width * scale_) + 2;  // slightly larger
}

void ScanOctree::reset_buffer() {
  id_buffer_.assign(width_ * width_, -1);
  z_buffer_.assign(width_ * width_, std::numeric_limits<float>::max());
}

void ScanOctree::z_buffer(vector<int>& drop_flags,
    const OctreeParser& octree_in) {
  // reset the buffer
  reset_buffer();

  const int depth = octree_in.info().depth();
  const int num = octree_in.info().node_num(depth);

  const uintk* key = octree_in.key_cpu(depth);
  const int* children = octree_in.children_cpu(depth);
  for (int i = 0; i < num; ++i) {
    if (children[i] < 0) continue;  // empty node

    float normal[3];
    octree_in.node_normal(normal, i, depth);
    if (dot_prod(normal, z_) >= 0) continue;  // invisible node

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

void ScanOctree::generate_flags(vector<vector<int>>& drop_flags,
    const OctreeParser& octree_in) {
  int depth = octree_in.info().depth();
  int depth_full = octree_in.info().full_layer();
  for (int d = 0; d < depth_full; ++d) {
    // keep all the nodes whose depth are smaller than depth_full
    drop_flags[d].assign(1ull << (3 * d), 0);
  }
  for (int d = depth - 1; d >= depth_full; --d) {
    int num = octree_in.info().node_num(d);
    drop_flags[d].assign(num, 1);
    const int* child_d = octree_in.children_cpu(d);
    for (int i = 0; i < num; ++i) {
      int j = child_d[i];
      if (j < 0) continue;  // empty node

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

void ScanOctree::trim_octree(vector<char>& octree_buffer,
    const OctreeParser& octree_in,
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
  octree_buffer.resize(info_out.sizeof_octree());
  OctreeParser octree_out;
  octree_out.set_cpu(octree_buffer.data(), &info_out);

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
    const uintk* key_in = octree_in.key_cpu(d);
    int* child_out = octree_out.mutable_children_cpu(d);
    uintk* key_out = octree_out.mutable_key_cpu(d);
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
      const float* feature_in = octree_in.feature_cpu(d);
      float* feature_out = octree_out.mutable_feature_cpu(d);
      for (int i = 0, j = 0; i < num; ++i) {
        // the node is dropped or empty
        if (drop[i] == 1) continue;
        int t = child_in[i];

        for (int k = 8 * t; k < 8 * t + 8; ++k) {
          for (int c = 0; c < channel_feature; ++c) {
            feature_out[c * nnum_out + j] =
                drop_d[k] == 0 ? feature_in[c * nnum_in + k] : 0;
          }
          j++;
        }
      }
    }

    // copy label
    if ((location_label == -1 || d == depth) && channel_label != 0) {
      const float* label_in = octree_in.label_cpu(d);
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
      const float* split_in = octree_in.split_cpu(d);
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

void octree_dropout(vector<char>& octree_output, const string& octree_input,
    const int depth_dropout, const float threshold) {
  // generate the drop flag
  OctreeParser parser_in;
  parser_in.set_cpu(octree_input.c_str());
  int depth = parser_in.info().depth();
  vector<vector<uintk>> drop(depth + 1);
  // generate random flag for the level depth_dropout
  int nnum_d = parser_in.info().node_num(depth_dropout);
  drop[depth_dropout].resize(nnum_d, 0);
  std::default_random_engine generator(static_cast<unsigned>(time(nullptr)));
  std::bernoulli_distribution distribution(threshold);
  for (int i = 0; i < nnum_d; ++i) {
    drop[depth_dropout][i] = static_cast<unsigned>(distribution(generator));
  }
  for (int d = depth_dropout + 1; d <= depth; ++d) {
    int nnum_d = parser_in.info().node_num(d);
    int nnum_dp = parser_in.info().node_num(d - 1);
    const int* children_dp = parser_in.children_cpu(d - 1);
    drop[d].resize(nnum_d);
    for (int i = 0; i < nnum_dp; ++i) {
      int t = children_dp[i];
      if (t < 0) continue;  // continue if it has no children
      // assign the drop flag of a parent node to its children
      for (int j = 0; j < 8; ++j) {
        drop[d][t * 8 + j] = drop[d - 1][i];
      }
    }
  }

  // init output
  OctreeInfo info_output = parser_in.info();
  vector<int> node_num(depth + 1, 0);
  for (int d = 0; d <= depth; ++d) {
    if (d <= depth_dropout) {
      node_num[d] = parser_in.info().node_num(d);
    } else {
      int num = 0;
      for (auto v : drop[d]) {
        if (v == 0) num++;
      }
      node_num[d] = num;
    }
  }
  info_output.set_nnum(node_num.data());
  info_output.set_nnum_cum();
  info_output.set_ptr_dis();
  octree_output.resize(info_output.sizeof_octree());
  OctreeParser parser_out;
  parser_out.set_cpu(octree_output.data(), &info_output);

  // start dropout
  // from level 0 to depth_output
  int num = parser_in.info().node_num_cum(depth_dropout + 1);
  int channel_key = parser_in.info().channel(OctreeInfo::kKey);
  // CHECK_EQ(channel_key, 1) << "Currently the channel must be 1";
  std::copy_n(parser_in.key_cpu(0), num * channel_key, parser_out.mutable_key_cpu(0));
  std::copy_n(parser_in.children_cpu(0), num, parser_out.mutable_children_cpu(0));
  int channel_feature = parser_in.info().channel(OctreeInfo::kFeature);
  int location_feature = parser_in.info().locations(OctreeInfo::kFeature);
  if (location_feature == -1) {
    std::copy_n(parser_in.feature_cpu(0), num * channel_feature,
        parser_out.mutable_feature_cpu(0));
  }
  int channel_label = parser_in.info().channel(OctreeInfo::kLabel);
  int location_label = parser_in.info().locations(OctreeInfo::kLabel);
  if (location_label == -1) {
    std::copy_n(parser_in.label_cpu(0), num * channel_label,
        parser_out.mutable_label_cpu(0));
  }
  int channel_split = parser_in.info().channel(OctreeInfo::kSplit);
  int location_split = parser_in.info().locations(OctreeInfo::kSplit);
  if (location_split == -1) {
    std::copy_n(parser_in.split_cpu(0), num * channel_split,
        parser_out.mutable_split_cpu(0));
  }

  // from level depth_output+1 to depth
  vector<int> node_num_nempty(depth + 1, 0);
  for (int d = depth_dropout + 1; d <= depth; ++d) {
    int nnum_d = parser_in.info().node_num(d), id = 0;
    const int* child_src = parser_in.children_cpu(d);
    const uintk* key_src = parser_in.key_cpu(d);
    int* child_des = parser_out.mutable_children_cpu(d);
    uintk* key_des = parser_out.mutable_key_cpu(d);
    for (int i = 0, j = 0; i < nnum_d; ++i) {
      if (drop[d][i] == 0) {
        key_des[j] = key_src[i];
        int ch = child_src[i] < 0 ? child_src[i] : id++;
        child_des[j] = ch;
        ++j;
      }
    }
    node_num_nempty[d] = id;

    if (location_feature == -1 || d == depth) {
      int nnum_src = parser_out.info().node_num(d);
      const float* feature_src = parser_in.feature_cpu(d);
      float* feature_des = parser_out.mutable_feature_cpu(d);
      for (int i = 0, j = 0; i < nnum_d; ++i) {
        if (drop[d][i] == 0) {
          for (int c = 0; c < channel_feature; ++c) {
            feature_des[c * nnum_src + j] = feature_src[c * nnum_d + i];
          }
          ++j;
        }
      }
    }

    if ((location_label == -1 || d == depth) && channel_label != 0) {
      const float* label_src = parser_in.label_cpu(d);
      float* label_des = parser_out.mutable_label_cpu(d);
      for (int i = 0, j = 0; i < nnum_d; ++i) {
        if (drop[d][i] == 0) {
          label_des[j] = label_src[i];
          ++j;
        }
      }
    }

    if ((location_split == -1 || d == depth) && channel_split != 0) {
      const float* split_src = parser_in.split_cpu(d);
      float* split_des = parser_out.mutable_split_cpu(d);
      for (int i = 0, j = 0; i < nnum_d; ++i) {
        if (drop[d][i] == 0) {
          split_des[j] = split_src[i];
          ++j;
        }
      }
    }
  }

  // modify the children and node_num_nempty
  int id = 0;
  const int* child_src = parser_in.children_cpu(depth_dropout);
  int* child_des = parser_out.mutable_children_cpu(depth_dropout);
  for (int i = 0; i < node_num[depth_dropout]; ++i) {
    child_des[i] =
        (drop[depth_dropout][i] == 1 || child_src[i] < 0) ? child_src[i] : id++;
  }
  for (int d = 0; d < depth_dropout; ++d) {
    node_num_nempty[d] = parser_in.info().node_num_nempty(d);
  }
  node_num_nempty[depth_dropout] = id;  // !!! important
  parser_out.mutable_info().set_nempty(node_num_nempty.data());
}

void upgrade_key64(vector<char>& octree_out, const vector<char>& octree_in) {
  typedef typename KeyTrait<uint32>::uints uints_in;
  typedef typename KeyTrait<uintk>::uints uints_out;

  OctreeParser parser_in, parser_out;
  parser_in.set_cpu(octree_in.data());


  // modify the OctreeInfo, update the magic_str and ptr_dis
  OctreeInfo oct_info = parser_in.info();
  oct_info.set_magic_str(OctreeInfo::kMagicStr);
  oct_info.set_ptr_dis();

  // set OctreeInfo to the output
  octree_out.resize(oct_info.sizeof_octree());
  parser_out.set_cpu(octree_out.data(), &oct_info);

  // copy key
  int num = oct_info.total_nnum();
  uintk* key_out = parser_out.mutable_key_cpu(0);
  const uint32* key_in = reinterpret_cast<const uint32*>(parser_in.key_cpu(0));
  for (int i = 0; i < num; ++i) {
    if (oct_info.is_key2xyz()) {
      uints_out* ptr_out = reinterpret_cast<uints_out*>(key_out + i);
      const uints_in* ptr_in = reinterpret_cast<const uints_in*>(key_in + i);
      for (int j = 0; j < 4; ++j) {
        ptr_out[j] = static_cast<uints_out>(ptr_in[j]);
      }
    } else {
      key_out[i] = static_cast<uintk>(key_in[i]);
    }
  }

  // copy children
  std::copy_n(parser_in.children_cpu(0), num, parser_out.mutable_children_cpu(0));

  // copy data
  int depth = oct_info.depth();
  int feature_channel = oct_info.channel(OctreeInfo::kFeature);
  int feature_location = oct_info.locations(OctreeInfo::kFeature);
  int depth_start = feature_location == depth ? depth : 0;
  for (int d = depth_start; d < depth + 1; ++d) {
    if (!oct_info.has_property(OctreeInfo::kFeature)) break;
    int n = oct_info.node_num(d) * feature_channel;
    std::copy_n(parser_in.feature_cpu(d), n, parser_out.mutable_feature_cpu(d));
  }

  // copy label
  int label_location = oct_info.locations(OctreeInfo::kLabel);
  depth_start = label_location == depth ? depth : 0;
  for (int d = depth_start; d < depth + 1; ++d) {
    if (!oct_info.has_property(OctreeInfo::kLabel)) break;
    int n = oct_info.node_num(d);
    std::copy_n(parser_in.label_cpu(d), n, parser_out.mutable_label_cpu(d));
  }

  // copy split label
  int split_location = oct_info.locations(OctreeInfo::kSplit);
  depth_start = split_location == depth ? depth : 0;
  for (int d = depth_start; d < depth + 1; ++d) {
    if (!oct_info.has_property(OctreeInfo::kSplit)) break;
    int n = oct_info.node_num(d);
    std::copy_n(parser_in.split_cpu(d), n, parser_out.mutable_split_cpu(d));
  }
}
