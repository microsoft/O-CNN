// todo: optimize the coding

#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

#include "filenames.h"
#include "octree.h"
#include "marching_cube.h"
#include "types.h"

using namespace std;

int depth_start = 4;
int signal_channel = 4;
int segmentation = 0;
int split_label = 0;

//float threshold = 0.002f;   // about 2.6 degree
float threshold = 0.1f;      // about 5.7 degree
float threshold_distance = 0.866; // sqrtf(3.0f) * 0.5f;
//float threshold = 0.02f;    // about 8.1 degree
//float threshold = 0.05f;    // about 12.8 degree
//float threshold = 0.2f;     // about 12.8 degree


void covered_depth_nodes(vector<vector<int>>& dnum_, vector<vector<int>>& didx_,
    const int* children_, const int* node_num_, const int* node_num_accu_, const int depth_) {
  // init
  didx_.resize(depth_ + 1);
  dnum_.resize(depth_ + 1);

  //layer-depth_
  int nnum = node_num_[depth_];
  dnum_[depth_].resize(nnum, 1);
  didx_[depth_].resize(nnum);
  for (int i = 0; i < nnum; ++i) {
    didx_[depth_][i] = i;
  }

  // layer-(depth_-1)
  nnum = node_num_[depth_ - 1];
  dnum_[depth_ - 1].resize(nnum, 0);
  didx_[depth_ - 1].resize(nnum, -1);
  const int* children_d = children_ + node_num_accu_[depth_ - 1];
  for (int i = 0; i < nnum; ++i) {
    int t = children_d[i];
    if (t != -1) {
      dnum_[depth_ - 1][i] = 8;
      didx_[depth_ - 1][i] = t * 8;
    }
  }

  // layer-(depth-2) to layer-0
  for (int d = depth_ - 2; d >= 0; --d) {
    nnum = node_num_[d];
    dnum_[d].resize(nnum, 0);
    didx_[d].resize(nnum, -1);
    const int* children_d = children_ + node_num_accu_[d];
    for (int i = 0; i < nnum; ++i) {
      int t = children_d[i];
      if (t != -1) {
        t *= 8;
        for (int j = 0; j < 8; ++j) {
          dnum_[d][i] += dnum_[d + 1][t + j];
        }
        for (int j = 0; j < 8; ++j) {
          if (didx_[d + 1][t + j] != -1) {
            didx_[d][i] = didx_[d + 1][t + j];
            break;
          }
        }
      }
    }
  }
}


void adaptive_octree(vector<char>& octree_output, const vector<char>& octree_input,
    const int depth_output) {
  /// const
  typedef typename KeyTrait<uintk>::uints uints;
  const float mul = sqrtf(3.0f) / 2.0f;
  const float imul = 2.0f / sqrtf(3.0f);
  //const int signal_channel = 4;

  /// parse the octree file
  Octree octree_in;
  octree_in.set_octree(octree_input.data(), octree_input.size());
  bool is_key2xyz = octree_in.info().is_key2xyz(); // !!! must be true
  int depth = octree_in.info().depth();
  int full_layer = octree_in.info().full_layer();
  int total_node_num = octree_in.info().total_nnum();
  int final_node_num = octree_in.info().node_num(depth);
  vector<int> nnum_vec(depth + 1, 0), nnum_accu_vec(depth + 2, 0);
  for (int d = 0; d < depth + 1; ++d) {
    nnum_vec[d] = octree_in.info().node_num(d);
    nnum_accu_vec[d] = octree_in.info().node_num_cum(d);
  }
  nnum_accu_vec[depth + 1] = octree_in.info().node_num_cum(depth + 1);
  const int* node_num = nnum_vec.data();
  const int* node_num_accu = nnum_accu_vec.data();
  const uintk* key = octree_in.key_cpu(0);
  const int* children = octree_in.children_cpu(0);
  const float* data = octree_in.feature_cpu(0);
  const float* normal_ptr = data; // !!! channel x n
  const float* dis_ptr = normal_ptr + 3 * final_node_num;
  const float* label_ptr = octree_in.label_cpu(0);

  /// precompute the nodes in the depth layer covered by each octree node
  vector<vector<int>> dnum_, didx_;
  covered_depth_nodes(dnum_, didx_, children, node_num, node_num_accu, depth);

  /// precompute the points in the finest level
  vector<float> pt_depth; // !!! n x channel
  if (signal_channel == 4) {
    pt_depth.resize(3 * final_node_num);
    const uintk* key_depth = key + node_num_accu[depth];
    for (int i = 0; i < final_node_num; ++i) {
      const uints* pt = reinterpret_cast<const uints*>(key_depth + i);
      for (int c = 0; c < 3; ++c) {
        float nc = normal_ptr[c * final_node_num + i];
        pt_depth[i * 3 + c] = static_cast<float>(pt[c]) + 0.5f + dis_ptr[i] * nc * mul;
      }
    }
  }

  /// the average normal & displacement and normal variation
  vector<vector<float> > normal_avg(depth + 1), normal_err(depth + 1), distance_err(depth + 1);
  vector<vector<float> > label_avg(depth + 1);
  int nlabel = 0;
  if (segmentation != 0) {
    nlabel = *std::max_element(label_ptr, label_ptr + final_node_num) + 1;
  }

  // initialization
  for (int d = 0; d <= depth; ++d) {
    normal_avg[d].resize(signal_channel * node_num[d], 0.0f);
    normal_err[d].resize(node_num[d], 5.0f); // !!! initialized as 5.0f
    distance_err[d].resize(node_num[d], 5.0e10f);
    if (segmentation != 0) {
      label_avg[d].resize(node_num[d], -1);
    }
  }

  // for the depth layer
  memcpy(normal_avg[depth].data(), normal_ptr, normal_avg[depth].size() * sizeof(float));
  if (segmentation != 0) {
    memcpy(label_avg[depth].data(), label_ptr, label_avg[depth].size() * sizeof(float));
  }

  // for the other layers
  for (int d = depth - 1; d > full_layer; --d) {
    vector<int>& dnum = dnum_[d];
    vector<int>& didx = didx_[d];
    const uintk* key_d = key + node_num_accu[d];
    const int* children_d = children + node_num_accu[d];
    const int* children_depth = children + node_num_accu[depth];
    const float scale = static_cast<float>(1 << (depth - d));

    int nnum = node_num[d];
    vector<float>& normal_d = normal_avg[d];
    vector<float>& normal_err_d = normal_err[d];
    vector<float>& distance_err_d = distance_err[d];

    vector<float>& label_d = label_avg[d];
    for (int i = 0; i < nnum; ++i) {
      if (children_d[i] == -1) continue;

      // average the normal and projection point
      float count = 0.0f;
      float pt_avg[3] = { 0.0f, 0.0f, 0.0f }, dis_avg = 0.0f;
      float n_avg[3] = { 0.0f, 0.0f, 0.0f };
      vector<int> l_avg(nlabel, 0);
      for (int j = didx[i]; j < didx[i] + dnum[i]; ++j) {
        if (children_depth[j] == -1)  continue;
        count += 1.0f;
        for (int c = 0; c < 3; ++c) {
          float nc = normal_ptr[c * final_node_num + j];
          n_avg[c] += nc;
          if (signal_channel == 4) {
            pt_avg[c] += pt_depth[3 * j + c];
          }
        }
        if (segmentation != 0) {
          l_avg[label_ptr[j]] += 1;
        }
      }


      float len = 1.0e-30f;
      for (int c = 0; c < 3; ++c) len += n_avg[c] * n_avg[c];
      len = sqrtf(len);

      float pt_base[3];
      const uints* pt = reinterpret_cast<const uints*>(key_d + i);
      for (int c = 0; c < 3; ++c) {
        n_avg[c] /= len;
        if (signal_channel == 4) {
          pt_avg[c] /= count * scale;   // !!! note the scale
          pt_base[c] = static_cast<float>(pt[c]);
          float fract_part = pt_avg[c] - pt_base[c];
          dis_avg += (fract_part - 0.5f) * n_avg[c];
        }
      }

      // === version 1
      // the normal error
      float nm_err = 0.0f;
      for (int j = didx[i]; j < didx[i] + dnum[i]; ++j) {
        if (children_depth[j] == -1)  continue;
        for (int c = 0; c < 3; ++c) {
          float tmp = normal_ptr[c * final_node_num + j] - n_avg[c];
          nm_err += tmp * tmp;
        }
      }
      nm_err /= count;

      // output
      normal_d[i] = n_avg[0];
      normal_d[1 * nnum + i] = n_avg[1];
      normal_d[2 * nnum + i] = n_avg[2];
      if (signal_channel == 4) {
        normal_d[3 * nnum + i] = dis_avg * imul; // IMPORTANT: RESCALE
      }
      normal_err_d[i] = nm_err;
      if (segmentation != 0) {
        label_d[i] = 0;
        for (int j = 1, v = 0; j < nlabel; ++j) {
          if (l_avg[j] > v) {
            v = l_avg[j];
            label_d[i] = j;
          }
        }
      }

      // === version 2
      if (signal_channel != 4) {
        distance_err_d[i] = 0;
        continue;
      }
      // the error from the original geometry to the averaged geometry
      float distance_max1 = -1;
      // !!! note the scale
      float pt_avg1[3] = { pt_avg[0] * scale, pt_avg[1] * scale, pt_avg[2] * scale };
      for (int j = didx[i]; j < didx[i] + dnum[i]; ++j) {
        if (children_depth[j] == -1)  continue;
        float dis = 0.0f;
        for (int c = 0; c < 3; ++c) {
          dis += (pt_depth[3 * j + c] - pt_avg1[c]) * n_avg[c];
        }
        dis = fabsf(dis);
        if (dis > distance_max1) distance_max1 = dis;
      }

      // the error from the averaged geometry to the original geometry
      float distance_max2 = -1;
      vector<float> vtx;
      intersect_cube(vtx, pt_avg, pt_base, n_avg);
      if (vtx.empty()) distance_max2 = 5.0e10f; // !!! the degenerated case, ||n_avg|| == 0
      for (auto& v : vtx) v *= scale;           // !!! note the scale
      for (int k = 0; k < vtx.size() / 3; ++k) {

        // min
        float distance_min = 1.0e30f;
        for (int j = didx[i]; j < didx[i] + dnum[i]; ++j) {
          if (children_depth[j] == -1)  continue;
          float dis = 0.0f;
          for (int c = 0; c < 3; ++c) {
            float ptc = pt_depth[3 * j + c] - vtx[3 * k + c];
            dis += ptc * ptc;
          }
          dis = sqrtf(dis);
          if (dis < distance_min) distance_min = dis;
        }

        // max
        if (distance_min > distance_max2) distance_max2 = distance_min;
      }

      distance_err_d[i] = std::max<float>(distance_max2, distance_max1);
    }
  }

  /// trim the octree according to normal_var
  vector<vector<float> > data_output(depth + 1), label_output(depth + 1);
  vector<vector<uintk> > key_output(depth + 1);
  vector<vector<int> > children_output(depth + 1), drop(depth + 1);
  for (int d = 0; d <= depth; ++d) {
    drop[d].resize(node_num[d], 0);  // 1 means dropping the sub-tree
  }
  for (int d = depth_output; d <= depth; ++d) {
    int nnum_d = node_num[d];
    int nnum_dp = node_num[d - 1];
    vector<float>& normal_err_d = normal_err[d];
    vector<float>& dist_err_d = distance_err[d];
    const uintk* key_d = key + node_num_accu[d];
    const int* children_d = children + node_num_accu[d];
    const int* children_dp = children + node_num_accu[d - 1];
    vector<int>& drop_d = drop[d];
    vector<int>& drop_dp = drop[d - 1];

    // generate the drop flag
    bool all_drop = true;
    for (int i = 0; i < nnum_dp; ++i) {
      int t = children_dp[i];
      if (t == -1) continue;
      for (int j = 0; j < 8; ++j) {
        int idx = t * 8 + j;
        drop_d[idx] = drop_dp[i] == 1 ||
            (normal_err_d[idx] < threshold && dist_err_d[idx] < threshold_distance);
        //drop_d[idx] = drop_dp[i] == 1 ||dist_err_d[idx] < thredhold_distance;
        if (all_drop && children_d[idx] != -1) {
          all_drop = drop_d[idx] == 1;
        }
      }
    }

    // make sure that there is at least one octree node in each layer
    if (all_drop) {
      int max_idx = 0;
      float max_var = -1.0f;
      for (int i = 0; i < nnum_dp; ++i) {
        int t = children_dp[i];
        if (t == -1 || drop_dp[i] == 1) continue;
        for (int j = 0; j < 8; ++j) {
          int idx = t * 8 + j;
          if (children_d[idx] != -1 && normal_err_d[idx] > max_var) {
            max_var = normal_err_d[idx];
            max_idx = idx;
          }
        }
      }
      drop_d[max_idx] = 0;
    }

    for (int i = 0, id = 0; i < nnum_dp; ++i) {
      int t = children_dp[i];
      if (t == -1) continue;
      for (int j = 0; j < 8; ++j) {
        int idx = t * 8 + j;
        if (drop_dp[i] == 0) {
          key_output[d].push_back(key_d[idx]);

          int ch = (drop_d[idx] == 0 && children_d[idx] != -1) ? id++ : -1;
          children_output[d].push_back(ch);

          for (int c = 0; c < signal_channel; ++c) {
            data_output[d].push_back(normal_avg[d][c * nnum_d + idx]);
          }

          if (segmentation != 0) {
            label_output[d].push_back(label_avg[d][idx]);
          }
        }
      }
    }

    // transpose data
    int num = key_output[d].size();
    vector<float> data_buffer(num * signal_channel);
    for (int i = 0; i < num; ++i) {
      for (int c = 0; c < signal_channel; ++c) {
        data_buffer[c * num + i] = data_output[d][i * signal_channel + c];
      }
    }
    data_output[d].swap(data_buffer);
  }

  /// output
  OctreeInfo info_out = octree_in.info();
  info_out.set_adaptive(true);
  info_out.set_threshold_dist(threshold_distance);
  info_out.set_threshold_normal(threshold);
  if (split_label) {
    info_out.set_property(OctreeInfo::kSplit, 1, -1);
  } else {
    info_out.set_property(OctreeInfo::kSplit, 0, 0);
  }
  info_out.set_property(OctreeInfo::kFeature, signal_channel, -1);
  if (segmentation) info_out.set_property(OctreeInfo::kLabel, 1, -1);
  //vector<int> nnum_nempty_vec(depth + 1);
  for (int d = 0; d <= depth; ++d) {
    nnum_vec[d] = d <= depth_output ? octree_in.info().node_num(d) : key_output[d].size();
  }
  info_out.set_nnum(nnum_vec.data());
  info_out.set_nnum_cum();
  info_out.set_ptr_dis();

  // copy OctreeInfo
  Octree octree_out;
  octree_out.resize_octree(info_out.sizeof_octree());
  octree_out.mutable_info() = info_out;
  copy(key, key + node_num_accu[depth_output], octree_out.mutable_key_cpu(0));
  for (int d = depth_output; d <= depth; ++d) {
    copy(key_output[d].begin(), key_output[d].end(), octree_out.mutable_key_cpu(d));
  }
  copy(children, children + node_num_accu[depth_output], octree_out.mutable_children_cpu(0));
  for (int d = depth_output; d <= depth; ++d) {
    copy(children_output[d].begin(), children_output[d].end(), octree_out.mutable_children_cpu(d));
  }
  for (int d = 0; d <= depth; ++d) {
    vector<float>& normal_tmp = d < depth_output ? normal_avg[d] : data_output[d];
    copy(normal_tmp.begin(), normal_tmp.end(), octree_out.mutable_feature_cpu(d));
  }
  if (segmentation != 0) {
    for (int d = 0; d <= depth; ++d) {
      vector<float>& label_tmp = d < depth_output ? label_avg[d] : label_output[d];
      copy(label_tmp.begin(), label_tmp.end(), octree_out.mutable_label_cpu(d));
    }
  }
  // update nnum_nempty
  for (int d = 0; d <= depth; ++d) {
    // find the last element which is not equal to -1
    int nnum_nempty = 0;
    const int* children_d = octree_out.children_cpu(d);
    for (int i = octree_out.info().node_num(d) - 1; i >= 0; i--) {
      if (children_d[i] != -1) {
        nnum_nempty = children_d[i] + 1;
        break;
      }
    }
    nnum_vec[d] = nnum_nempty;
  }
  octree_out.mutable_info().set_nempty(nnum_vec.data());

  if (split_label != 0) {
    // generate split label according to the children_
    vector<vector<float> > split_output(depth + 1);
    for (int d = 0; d <= depth; ++d) {
      int nnum_d = octree_out.info().node_num(d);
      vector<float>& split_d = split_output[d];
      split_d.resize(nnum_d, 1); // initialize as 1
      const int* children_d = d < depth_output ?
          children + node_num_accu[d] : children_output[d].data();
      vector<float>& data_d = data_output[d];
      for (int i = 0; i < nnum_d; ++i) {
        if (children_d[i] == -1) {
          split_d[i] = 0;
          if (d >= depth_output) {
            float t = fabsf(data_d[i]) + fabsf(data_d[nnum_d + i]) + fabsf(data_d[nnum_d * 2 + i]);
            if (t != 0) split_d[i] = 2;
          }
        }
      }
    }
    for (int d = 0; d <= depth; ++d) {
      copy(split_output[d].begin(), split_output[d].end(), octree_out.mutable_split_cpu(d));
    }
  }

  octree_output = octree_out.buffer();
}


void adaptive_octree(const string& filename, const string& filename_output) {
  // read octree
  ifstream infile(filename, ios::binary);
  infile.seekg(0, infile.end);
  int len = infile.tellg();
  infile.seekg(0, infile.beg);
  vector<char> octree(len, 0);
  infile.read(octree.data(), len);
  infile.close();

  vector<char> octree_output;
  adaptive_octree(octree_output, octree, depth_start);

  // save octree
  ofstream outfile(filename_output, ios::binary);
  outfile.write(octree_output.data(), octree_output.size());
  outfile.close();
}


int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "Usage: AdaptiveOctree.exe <input file path> <output file path> "
        << "[depth] [signal_channel] [segmentation] [split_label]" << endl;
    return 0;
  }

  string input_file_path(argv[1]);
  string output_file_path(argv[2]);
  mkdir(output_file_path.c_str());
  if (argc > 3) depth_start = atoi(argv[3]);
  if (argc > 4) signal_channel = atoi(argv[4]);
  if (argc > 5) segmentation = atoi(argv[5]);
  if (argc > 6) split_label = atoi(argv[6]);

  vector<string> all_files;
  get_all_filenames(all_files, input_file_path + "\\*.octree");

  //#pragma omp parallel for
  for (int i = 0; i < all_files.size(); ++i) {
    string filename = extract_filename(all_files[i]);
    string filename_ouput = output_file_path + filename + "_output.octree";

    adaptive_octree(all_files[i], filename_ouput);

    cout << filename + " done!\n";
  }

  return 0;

}
