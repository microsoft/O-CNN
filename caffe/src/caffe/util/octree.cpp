#include <boost/thread.hpp>
#include <unordered_map>
#include <omp.h>
#include "caffe/util/octree.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// Make sure each thread can have different values.
static boost::thread_specific_ptr<Octree> thread_octree_instance_;

Octree& Octree::Get() {
  if (!thread_octree_instance_.get()) {
    thread_octree_instance_.reset(new Octree());
  }
  return *(thread_octree_instance_.get());
}

shared_ptr<Blob<float> > Octree::get_workspace(float, int id) {
  vector<int> shape{ 1 };
  auto& workspace = Get().workspace_;
  if (id + 1 > workspace.size()) workspace.resize(id + 1, nullptr);
  if (!workspace[id].get()) workspace[id].reset(new Blob<float>(shape));
  return workspace[id];
}

shared_ptr<Blob<double> > Octree::get_workspace(double, int id) {
  vector<int> shape{ 1 };
  auto& workspaced = Get().workspaced_;
  if (id + 1 > workspaced.size()) workspaced.resize(id + 1, nullptr);
  if (!workspaced[id].get()) workspaced[id].reset(new Blob<double>(shape));
  return workspaced[id];
}

void Octree::init_neigh_index() {
  const vector<std::pair<string, int> > kernel_type{
    { "333", 0 }, { "111", 1 }, { "222", 2 },
    { "311", 3 }, { "131", 4 }, { "113", 5 },
    { "331", 6 }, { "313", 7 }, { "133", 8 } };

  const vector<vector<int> > vec{ {} /* 333 */, { 13 } /* 111 */,
    { 13, 14, 16, 17, 22, 23, 25, 26 } /* 222 */,
    {  4, 13, 22 } /* 311 */,
    { 10, 13, 16 } /* 131 */,
    { 12, 13, 14 } /* 113 */,
    { 1,  4,  7, 10, 13, 16, 19, 22, 25 } /* 331 */,
    { 3,  4,  5, 12, 13, 14, 21, 22, 23 } /* 313 */,
    { 9, 10, 11, 12, 13, 14, 15, 16, 17 } /* 133 */ };

  // init
  ni_map_.insert(kernel_type.begin(), kernel_type.end());
  ni_.resize(kernel_type.size());

  // ni for kernel_size=333
  ni_[0].reset(new Blob<int>(vector<int> { 216 }));
  int* ni3 = ni_[0]->mutable_cpu_data();
  int id = 0;
  for (int ijk = 0; ijk < 8; ++ijk) {
    for (int xyz = 0; xyz < 27; ++xyz) {
      int k = ijk % 2, p = ijk / 2;
      int j = p % 2,   i = p / 2;

      int z = xyz % 3, q = xyz / 3;
      int y = q % 3,   x = q / 3;

      ni3[id++] = ((x + i) << 4) | ((y + j) << 2) | (z + k);
    }
  }

  // ni for other kernel_sizes
  for (int k = 1; k < kernel_type.size(); ++k) {
    int sz = vec[k].size();
    ni_[k].reset(new Blob<int>(vector<int> { 8 * sz }));
    int* ni = ni_[k]->mutable_cpu_data();
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < sz; ++j) {
        ni[i * sz + j] = ni3[i * 27 + vec[k][j]];
      }
    }
  }

  // init the array parent & displacement
  id = 0;
  int tmp[64];
  displacement_.Reshape(vector<int> { 64 });
  int* dis_ptr = displacement_.mutable_cpu_data();
  for (int x = 1; x < 5; ++x) {
    for (int y = 1; y < 5; ++y) {
      for (int z = 1; z < 5; ++z) {
        int x1 = x / 2;
        int xb = x % 2;
        int y1 = y / 2;
        int yb = y % 2;
        int z1 = z / 2;
        int zb = z % 2;

        tmp[id] = x1 * 9 + y1 * 3 + z1;
        dis_ptr[id] = (xb << 2) | (yb << 1) | zb;
        id++;
      }
    }
  }

  parent_.Reshape(vector<int> { 512 });
  int* parent_ptr = parent_.mutable_cpu_data();
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 64; ++j) {
      parent_ptr[i * 64 + j] = ni3[i * 27 + tmp[j]];
    }
  }
}

shared_ptr<Blob<int> > Octree::get_ni(const vector<int>& kernel_size) {
  CHECK_EQ(kernel_size.size(), 3);
  string key;
  for (auto i : kernel_size) key += std::to_string(i);
  auto it = Get().ni_map_.find(key);
  CHECK(!(it == Get().ni_map_.end())) << "Unsupported kernel_size";

  return Get().ni_[it->second];
}

namespace octree {

template<typename Dtype>
void pad_forward_cpu(Dtype* Y, const int Hy,
    const int Cy, const Dtype* X, const int Hx, const int* label) {
  // Note: Cx == Cy
  for (int c = 0; c < Cy; ++c) {
    for (int h = 0; h < Hy; ++h) {
      Y[c * Hy + h] = label[h] == -1 ? Dtype(0) : X[c * Hx + label[h]];
    }
  }
}
template<typename Dtype>
void pad_backward_cpu(Dtype* X, const int Hx,
    const int Cx, const Dtype* Y, const int Hy, const int* label) {
  // Note: Cx == Cy
  for (int c = 0; c < Cx; ++c) {
    for (int h = 0; h < Hy; ++h) {
      if (label[h] != -1) {
        X[c * Hx + label[h]] = Y[c * Hy + h];
      }
    }
  }
}

template <typename Dtype>
void octree2col_cpu(Dtype* data_col, const Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n) {
  // height : the ideal height of workspace
  // height_col : the actual height of workspace
  const int octree_h = height << 3 * (stride - 1);
  const int kernel = kernel_sdim;
  for (int c = 0; c < channel; ++c) {
    for (int k = 0; k < kernel; ++k) {
      int h_start = n * height_col;
      int i_start = (c * kernel + k) * height_col - h_start;
      for (int h = h_start; h < h_start + height_col; ++h) {
        if (h >= height) {
          data_col[i_start + h] = Dtype(0);
          continue;
        }
        const int index = stride == 2 ? (h << 6) + ni[k] :
            (h >> 3 << 6) + ni[(h % 8) * kernel + k];
        const int p = neigh[index];
        data_col[i_start + h] =
            p == -1 ? Dtype(0) : data_octree[c * octree_h + p];
      }
    }
  }
}

template <typename Dtype>
void col2octree_cpu(const Dtype* data_col, Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n) {
  // height : the ideal height of workspace
  // height_col : the actual height of workspace
  const int octree_h = height << 3 * (stride - 1);
  const int kernel = kernel_sdim;
  // set data_octree to zero ONCE when n ==0
  if (n == 0) caffe_set(channel * octree_h, Dtype(0), data_octree);
  for (int c = 0; c < channel; ++c) {
    for (int k = 0; k < kernel; ++k) {
      int h_start = n * height_col;
      int i_start = (c * kernel + k) * height_col - h_start;
      for (int h = h_start; h < h_start + height_col; ++h) {
        if (h >= height) continue;
        const int index = stride == 2 ? (h << 6) + ni[k] :
            (h >> 3 << 6) + ni[(h % 8) * kernel + k];
        const int p = neigh[index];
        if (p != -1)
          data_octree[c * octree_h + p] += data_col[i_start + h];
      }
    }
  }
}

void generate_key_cpu(unsigned int* key_split, const unsigned int* key,
    const int* children, const int node_num) {
  typedef unsigned char ubyte;
  for (int i = 0; i < node_num; ++i) {
    int label = children[i];
    if (label == -1) continue;
    const ubyte* k0 = (const ubyte*)(key + i);
    for (ubyte j = 0; j < 8; ++j) {
      ubyte* k1 = (ubyte*)(key_split + 8 * label + j);
      k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
      k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
      k1[2] = (k0[2] << 1) | (j & 1);
      k1[3] = k0[3];
    }
  }
}

void generate_key_cpu(unsigned int* key, const int depth, const int batch_size) {
  int node_num = 1 << 3 * depth;
  for (int n = 0; n < batch_size; ++n) {
    for (int k = 0; k < node_num; ++k) {
      unsigned xyz = 0;
      unsigned char* ptr = (unsigned char*)(&xyz);
      for (int d = 0; d < depth; d++) {
        ptr[0] |= (k & (1 << (3 * d + 2))) >> (2 * d + 2);
        ptr[1] |= (k & (1 << (3 * d + 1))) >> (2 * d + 1);
        ptr[2] |= (k & (1 << (3 * d + 0))) >> (2 * d + 0);
      }
      ptr[3] = n;
      key[n * node_num + k] = xyz;
    }
  }
}

// NOTE: !!! currently the depth should be less than 8
void xyz2key_cpu(unsigned int* key, const unsigned int* xyz, const int num, const int depth) {
  for (int i = 0; i < num; ++i) {
    unsigned int pt[3] = { 0, 0, 0 }, key_out = 0;
    const unsigned char* ptr = reinterpret_cast<const unsigned char*>(xyz + i);
    unsigned char* ptr_out = (unsigned char*)(&key_out);
    for (int j = 0; j < 3; ++j) {
      pt[j] = static_cast<unsigned int>(ptr[j]);
    }
    compute_key(key_out, pt, depth);
    ptr_out[3] = ptr[3];
    key[i] = key_out;
  }
}

template <typename Dtype>
void generate_label_cpu(int* label_data, int& top_h, const Dtype* bottom_data,
    const int bottom_h, const int mask) {
  top_h = 0;
  for (int i = 0; i < bottom_h; ++i) {
    label_data[i] = (mask == static_cast<int>(bottom_data[i])) ? top_h++ : -1;
  }
}

void calc_neigh_cpu(int* neigh_split, const int* neigh,
    const int* children, const int node_num) {
  const int* parent = Octree::get_parent_array().cpu_data();
  const int* dis = Octree::get_dis_array().cpu_data();

  #pragma omp parallel for
  for (int i = 0; i < node_num; ++i) {
    int l0 = children[i];
    if (l0 == -1) continue;
    const int* ngh0 = neigh + (i >> 3 << 6);
    const int* pi0 = parent + (i % 8) * 64;
    int* ngh1 = neigh_split + (l0 << 6);
    for (int j = 0; j < 64; ++j) {
      ngh1[j] = -1;
      int k = ngh0[pi0[j]];
      if (k != -1) {
        int l1 = children[k];
        if (l1 != -1) {
          ngh1[j] = (l1 << 3) + dis[j];
        }
      }
    }
  }
}

void calc_neigh_cpu(int* neigh, const int depth, const int batch_size) {
  typedef unsigned int uint32;

  uint32 node_num = 1 << 3 * depth;
  const uint32  bound = 1 << depth;
  for (uint32 n = 0; n < batch_size; ++n) {
    for (uint32 i = 0; i < node_num; i += 8) {
      // key to xyz
      uint32 x0 = 0, y0 = 0, z0 = 0;
      for (uint32 d = 0; d < depth; d++) {
        x0 |= (i & (1 << (3 * d + 2))) >> (2 * d + 2);
        y0 |= (i & (1 << (3 * d + 1))) >> (2 * d + 1);
        z0 |= (i & (1 << (3 * d + 0))) >> (2 * d + 0);
      }

      for (uint32 x = 0; x < 4; ++x) {
        for (uint32 y = 0; y < 4; ++y) {
          for (uint32 z = 0; z < 4; ++z) {
            uint32 x1 = x0 + x - 1;
            uint32 y1 = y0 + y - 1;
            uint32 z1 = z0 + z - 1;

            int v = -1;
            if ((x1 & bound) == 0 &&
                (y1 & bound) == 0 &&
                (z1 & bound) == 0) {
              uint32 key1 = 0;
              for (int d = 0; d < depth; d++) {
                uint32 mask = 1u << d;
                key1 |= ((x1 & mask) << (2 * d + 2)) |
                    ((y1 & mask) << (2 * d + 1)) |
                    ((z1 & mask) << (2 * d));
              }
              v = key1 + n * node_num;
            }

            uint32 xyz = (x << 4) | (y << 2) | z;
            neigh[xyz + i * 8 + n * node_num * 8] = v;
          }
        }
      }
    }
  }
}

void calc_neighbor(int* neigh, const unsigned* key, const int node_num,
    const int displacement) {
  typedef unsigned char ubyte;

  // build hash table
  vector<std::pair<unsigned, int>> entries(node_num);
  for (int id = 0; id < node_num; ++id) {
    // ignore the root node
    entries[id] = std::make_pair(key[id], id + displacement);
  }
  std::unordered_map<unsigned, int> hash_table(entries.begin(), entries.end());

  // calc neighborhood
  for (int id = 0; id < node_num; id += 8) {
    // the neighborhood volume
    int* ngh = neigh + id * 8;
    const ubyte* k0 = (const ubyte*)(key + id);
    // currently the maximize octree depth is 8
    ubyte k1[4] = { 0, 0, 0, k0[3] };
    //const ubyte bound = (1 << k0[3]) - 2;
    for (ubyte x = 0; x < 4; ++x) {
      k1[0] = k0[0] + x - 1;
      for (ubyte y = 0; y < 4; ++y) {
        k1[1] = k0[1] + y - 1;
        for (ubyte z = 0; z < 4; ++z) {
          k1[2] = k0[2] + z - 1;

          // find
          unsigned* k2 = reinterpret_cast<unsigned*>(k1);
          auto rst = hash_table.find(*k2);
          ubyte i = (x << 4) | (y << 2) | z;
          if (rst != hash_table.end()) {
            ngh[i] = rst->second;
          } else {
            ngh[i] = -1;
          }
        }
      }
    }
  }
}

inline void compute_key(unsigned int& key, const unsigned int* pt, const int depth) {
  key = 0;
  for (int i = 0; i < depth; i++) {
    int mask = 1u << i;
    for (int j = 0; j < 3; j++) {
      key |= (pt[j] & mask) << (2 * i + 2 - j);
    }
  }
}

inline void compute_pt(unsigned int* pt, const unsigned int& key, const int depth) {
  for (int i = 0; i < 3; pt[i++] = 0u);
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < 3; j++) {
      // bit mask
      int mask = 1u << (3 * i + 2 - j);
      // put the bit to position i
      pt[j] |= (key & mask) >> (2 * i + 2 - j);
    }
  }
}

void octree_dropout(vector<char>& octree_output, const string& octree_input,
    const int depth_dropout, const float threshold, const int channel) {
  // parse the octree file
  //OctreeParser octin(octree_input.data(), channel);
  //const int depth = *octin.depth_;
  //const int* nnum = octin.node_num_;
  //const int* nnum_accu = octin.node_num_accu_;

  //// generate random num
  //int nnum_d = nnum[depth_dropout];
  //vector<int> dropout(nnum_d, 0), dropout_d;
  //caffe_rng_bernoulli(nnum_d, threshold, dropout.data());

  //// start dropout
  //vector<float> data_output;
  //vector<vector<int> > key_output(depth + 1), children_output(depth + 1);
  //key_output[depth_dropout].assign(octin.key_ + nnum_accu[depth_dropout],
  //    octin.key_ + nnum_accu[depth_dropout + 1]);
  //children_output[depth_dropout].resize(nnum_d);
  //const int* children_d = octin.children_ + nnum_accu[depth_dropout];
  //for (int i = 0, id = 0; i < nnum_d; ++i) {
  //  children_output[depth_dropout][i] =
  //      (dropout[i] == 0 && children_d[i] != -1) ? id++ : -1;
  //}

  //for (int d = depth_dropout + 1; d <= depth; ++d) {
  //  // generate random drop flag for current octree level
  //  int nnum_d = nnum[d];
  //  int nnum_dp = nnum[d - 1];
  //  const int* children_dp = octin.children_ + nnum_accu[d - 1];
  //  dropout_d.resize(nnum_d);
  //  for (int i = 0; i < nnum_dp; ++i) {
  //    int t = children_dp[i];
  //    if (t == -1) continue;
  //    for (int j = 0; j < 8; ++j) {
  //      dropout_d[t * 8 + j] = dropout[i];
  //    }
  //  }

  //  // generate key & children
  //  key_output[d].reserve(nnum_d);
  //  children_output[d].reserve(nnum_d);
  //  const int* key_d = octin.key_ + octin.node_num_accu_[d];
  //  const int* children_d = octin.children_ + octin.node_num_accu_[d];
  //  for (int i = 0, id = 0; i < nnum_d; ++i) {
  //    if (dropout_d[i] == 0) {
  //      key_output[d].push_back(key_d[i]);

  //      int ch = children_d[i] == -1 ? -1 : id++;
  //      children_output[d].push_back(ch);
  //    }
  //  }

  //  // generate data
  //  if (d == depth) {
  //    int num = key_output[d].size();
  //    data_output.resize(channel * num);
  //    const float* normal = reinterpret_cast<const float*>(octin.signal_);
  //    for (int i = 0, id = 0; i < nnum_d; ++i) {
  //      if (dropout_d[i] == 0) {
  //        for (int c = 0; c < channel; ++c) {
  //          data_output[c * num + id] = normal[c * nnum_d + i];
  //        }
  //        id++;
  //      }
  //    }
  //  }

  //  // swap
  //  dropout.swap(dropout_d);
  //}

  //// node num
  //vector<int> nnum_output(nnum, nnum + depth + 1);
  //vector<int> nnum_accu_output(nnum_accu, nnum_accu + depth + 2);
  //for (int d = depth_dropout + 1; d <= depth; ++d) {
  //  nnum_output[d] = key_output[d].size();
  //  nnum_accu_output[d + 1] = nnum_accu_output[d] + nnum_output[d];
  //}
  //int final_nnum = nnum_output[depth];
  //int total_nnum = nnum_accu_output[depth + 1];

  //// split label
  //vector<int> split_label(total_nnum);
  //auto op = [](int x) { return x < 0 ? 0 : 1; };
  //auto it = std::transform(octin.children_,
  //        octin.children_ + nnum_accu[depth_dropout + 1], split_label.begin(), op);
  //for (int d = depth_dropout + 1; d <= depth; ++d) {
  //  it = std::transform(children_output[d].begin(),	children_output[d].end(), it, op);
  //}

  //// output
  //int sz = sizeof(int) * (2 * depth + 7 + 3 * total_nnum + channel * final_nnum);
  //octree_output.resize(sz);
  //int* octo = reinterpret_cast<int*>(octree_output.data());
  //octo[0] = total_nnum;
  //octo[1] = final_nnum;
  //octo[2] = depth;
  //octo[3] = *octin.full_layer_;
  //int* ptr = octo + 4;
  //ptr = std::copy(nnum_output.begin(), nnum_output.end(), ptr);
  //ptr = std::copy(nnum_accu_output.begin(), nnum_accu_output.end(), ptr);
  //ptr = std::copy_n(octin.key_, nnum_accu[depth_dropout], ptr);
  //for (int d = depth_dropout; d <= depth; ++d) {
  //  ptr = std::copy_n(key_output[d].begin(), key_output[d].size(), ptr);
  //}
  //ptr = std::copy_n(octin.children_, nnum_accu[depth_dropout], ptr);
  //for (int d = depth_dropout; d <= depth; ++d) {
  //  ptr = std::copy_n(children_output[d].begin(), children_output[d].size(), ptr);
  //}
  //memcpy(ptr, data_output.data(), sizeof(float)*data_output.size());
  //ptr += data_output.size();
  //std::copy(split_label.begin(), split_label.end(), ptr);
}

void aoctree_dropout(vector<char>& octree_output, const string& octree_input,
    const int depth_dropout, const int channel) {
  //// parse the octree file
  //OctreeParser octin(octree_input.data(), channel);
  //const int depth = *octin.depth_;
  //const int* nnum = octin.node_num_;
  //const int* nnum_accu = octin.node_num_accu_;
  //if (depth_dropout > depth) {
  //  octree_output.resize(octree_input.size());
  //  std::copy(octree_input.begin(), octree_input.end(), octree_output.begin());
  //  return;
  //}

  //// retain at least one octant in each level
  //vector<int> idx(depth + 2, 0);
  //for (int d = depth; d >= depth_dropout; --d) {
  //  int ci = idx[d + 1] >> 3;
  //  int nnum_d = nnum[d];
  //  const int* children_d = octin.children_ + nnum_accu[d];
  //  for (int i = 0; i < nnum_d; ++i) {
  //    if (children_d[i] == ci) {
  //      idx[d] = i;
  //      break;
  //    }
  //  }
  //}

  //// start dropout
  //vector<vector<int> > data_output(depth + 1);
  //vector<vector<int> > children_output(depth + 1);
  //vector<vector<int> > key_output(depth + 1);
  //int nnum_d = nnum[depth_dropout];
  //children_output[depth_dropout].resize(nnum_d, -1);
  //children_output[depth_dropout][idx[depth_dropout]] = 0;
  //for (int d = depth_dropout + 1; d <= depth; ++d) {
  //  int i = idx[d];
  //  int j = i % 8;
  //  key_output[d].resize(8, 0);
  //  const int* key_d = octin.key_ + nnum_accu[d];
  //  for (int k = 0, h = i >> 3 << 3; k < 8; ++k, ++h) {
  //    key_output[d][k] = key_d[h];
  //  }

  //  children_output[d].resize(8, -1);
  //  children_output[d][j] = 0;

  //  nnum_d = nnum[d];
  //  data_output[d].resize(8 * channel, 0);
  //  const int* data_d = octin.signal_ + channel * nnum_accu[d];
  //  for (int c = 0; c < channel; ++c) {
  //    data_output[d][c * 8 + j] = data_d[c * nnum_d + i];
  //  }
  //}

  //// node num
  //vector<int> nnum_output(nnum, nnum + depth + 1);
  //vector<int> nnum_accu_output(nnum_accu, nnum_accu + depth + 2);
  //for (int d = depth_dropout + 1; d <= depth; ++d) {
  //  nnum_output[d] = key_output[d].size();
  //  nnum_accu_output[d + 1] = nnum_accu_output[d] + nnum_output[d];
  //}
  //int final_nnum = nnum_output[depth];
  //int total_nnum = nnum_accu_output[depth + 1];

  //// output
  //int sz = sizeof(int) * (2 * depth + 7 + (3 + channel) * total_nnum);
  //octree_output.resize(sz);
  //int* octo = reinterpret_cast<int*>(octree_output.data());
  //octo[0] = total_nnum;
  //octo[1] = final_nnum;
  //octo[2] = depth;
  //octo[3] = *octin.full_layer_;
  //int* ptr = octo + 4;
  //ptr = std::copy(nnum_output.begin(), nnum_output.end(), ptr);
  //ptr = std::copy(nnum_accu_output.begin(), nnum_accu_output.end(), ptr);
  //ptr = std::copy_n(octin.key_, nnum_accu[depth_dropout + 1], ptr);
  //for (int d = depth_dropout + 1; d <= depth; ++d) {
  //  ptr = std::copy_n(key_output[d].begin(), key_output[d].size(), ptr);
  //}
  //ptr = std::copy_n(octin.children_, nnum_accu[depth_dropout], ptr);
  //for (int d = depth_dropout; d <= depth; ++d) {
  //  ptr = std::copy_n(children_output[d].begin(), children_output[d].size(), ptr);
  //}

  //ptr = std::copy_n(octin.signal_, nnum_accu[depth_dropout + 1] * channel, ptr);
  //for (int d = depth_dropout + 1; d <= depth; ++d) {
  //  ptr = std::copy_n(data_output[d].begin(), data_output[d].size(), ptr);
  //}
}


template<typename Dtype>
void merge_octrees(Blob<Dtype>& octree_output, const vector<vector<char> >& octree_buffer) {
  /// parse the input octrees
  int batch_size = octree_buffer.size();
  vector<OctreeParser> octree_parsers(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    octree_parsers[i].set_cpu(octree_buffer[i].data());
  }

  // get depth and full_layer information
  string err_msg;
  const int depth = octree_parsers[0].info().depth();
  const int full_layer = octree_parsers[0].info().full_layer();
  bool valid = octree_parsers[0].info().check_format(err_msg);
  CHECK(valid) << err_msg;
  for (int i = 1; i < batch_size; ++i) {
    valid = octree_parsers[i].info().check_format(err_msg);
    CHECK(valid) << err_msg;
    CHECK(octree_parsers[0].info().is_consistent(octree_parsers[i].info()))
        << "The formats of input octrees are not consistent, check the database";
  }

  /// get the node number information
  // node and non-empty node number in each octree
  int sz = (depth + 1) * batch_size;
  vector<int> nnum(sz), nnum_nempty(sz);
  for (int i = 0; i < batch_size; ++i) {
    for (int d = 0; d < depth + 1; ++d) {
      int p = i * (depth + 1) + d;
      nnum[p] = octree_parsers[i].info().node_num(d);
      nnum_nempty[p] = octree_parsers[i].info().node_num_nempty(d);
    }
  }

  // cumulative node and non-empty node number in each layers
  sz = (depth + 1) * (batch_size + 1);
  vector<int> nnum_cum_layer(sz), nnum_cum_nempty_layer(sz);
  for (int d = 0; d < depth + 1; ++d) {
    nnum_cum_layer[d] = 0;
    nnum_cum_nempty_layer[d] = 0;
    for (int i = 0; i < batch_size; ++i) {
      int p = i * (depth + 1) + d;
      int q = p + depth + 1;
      nnum_cum_layer[q] = nnum[p] + nnum_cum_layer[p];
      nnum_cum_nempty_layer[q] = nnum_nempty[p] + nnum_cum_nempty_layer[p];
    }
  }

  // cumulative node number for each octree
  sz = (depth + 1) * batch_size;
  vector<int> nnum_cum_octree(sz);
  for (int i = 0; i < batch_size; ++i) {
    nnum_cum_octree[i * (depth + 1)] = 0;
    for (int d = 0; d < depth; ++d) {
      int p = i * (depth + 1) + d;
      nnum_cum_octree[p + 1] = nnum_cum_octree[p] + nnum[p];
    }
  }

  // node and non-empty node number of the batch
  vector<int> nnum_batch(depth + 1), nnum_nempty_batch(depth + 1);
  for (int d = 0; d < depth + 1; ++d) {
    int p = batch_size * (depth + 1) + d;
    nnum_batch[d] = nnum_cum_layer[p];
    nnum_nempty_batch[d] = nnum_cum_nempty_layer[p];
  }

  // cumulative node number of the batch
  vector<int> nnum_cum_batch(depth + 2);
  nnum_cum_batch[0] = 0;
  for (int d = 0; d < depth + 1; ++d) {
    nnum_cum_batch[d + 1] = nnum_cum_batch[d] + nnum_batch[d];
  }


  /// set the octinfo
  OctreeInfo info_batch = octree_parsers[0].info();
  info_batch.set_batch_size(batch_size);
  // add the neighbor property
  const int kNeighChannel = 8;
  info_batch.set_property(OctreeInfo::kNeigh, kNeighChannel, -1);
  // update nodenumber
  info_batch.set_nnum(nnum_batch.data());
  info_batch.set_nempty(nnum_nempty_batch.data());
  info_batch.set_nnum_cum();
  info_batch.set_ptr_dis();
  valid = info_batch.check_format(err_msg);
  CHECK(valid) << err_msg;

  /// reshape the blobs
  sz = info_batch.sizeof_octree() / sizeof(Dtype) + 1;
  octree_output.Reshape(vector<int> {sz});
  OctreeParser octbatch_parser;
  octbatch_parser.set_cpu(octree_output.mutable_cpu_data(), &info_batch);


  /// set data
  // If the OpenMP is available, the code can be more concise.
  //omp_set_num_threads(8);
  //#pragma omp parallel for
  //for (int i = 0; i < batch_size; ++i) {
  auto worker = [&](int thread_id, int thread_num) {
    for (int i = thread_id; i < batch_size; i += thread_num) {
      // copy key
      // TODO: optimize! TODO: IF DEPTH > 8
      // the channel and location of key is 1 and -1 (todo: !!! channel 2 for deeper key)
      for (int d = 0; d < depth + 1; ++d) {
        if (!info_batch.has_property(OctreeInfo::kKey)) break;
        int p = i * (depth + 1) + d;
        unsigned int* des = octbatch_parser.mutable_key_cpu(d) + nnum_cum_layer[p];
        const unsigned int* src = octree_parsers[i].key_cpu(d);
        for (int j = 0; j < nnum[p]; ++j) {
          des[j] = src[j];
          // !!! todo: deal with octree depth > 8
          unsigned char* ptr = reinterpret_cast<unsigned char*>(des + j);
          ptr[3] = i;
        }
      }

      // copy children
      // by default, the channel and location of children is 1 and -1,
      for (int d = 0; d < depth + 1; ++d) {
        if (!info_batch.has_property(OctreeInfo::kChild)) break;
        int p = i * (depth + 1) + d;
        int* des = octbatch_parser.mutable_children_cpu(d) + nnum_cum_layer[p];
        const int* src = octree_parsers[i].children_cpu(d);
        for (int j = 0; j < nnum[p]; ++j) {
          des[j] = -1 == src[j] ? src[j] : src[j] + nnum_cum_nempty_layer[p];
        }
      }

      // copy data: !NOTE! the type of signal is float!!!
      int feature_channel = info_batch.channel(OctreeInfo::kFeature);
      int feature_location = info_batch.locations(OctreeInfo::kFeature);
      int depth_start = feature_location == depth ? depth : 0;
      for (int d = depth_start; d < depth + 1; ++d) {
        if (!info_batch.has_property(OctreeInfo::kFeature)) break;
        int p = i * (depth + 1) + d;
        for (int c = 0; c < feature_channel; c++) {
          float* des = octbatch_parser.mutable_feature_cpu(d) + c * nnum_batch[d] + nnum_cum_layer[p];
          const float* src = octree_parsers[i].feature_cpu(d) + c * nnum[p];
          for (int j = 0; j < nnum[p]; ++j) { des[j] = src[j]; }
        }
      }

      // copy label: !NOTE! the type of label is float!!!
      int label_location = info_batch.locations(OctreeInfo::kLabel);
      depth_start = label_location == depth ? depth : 0;
      for (int d = depth_start; d < depth + 1; ++d) {
        if (!info_batch.has_property(OctreeInfo::kLabel)) break;
        int p = i * (depth + 1) + d;
        float* des = octbatch_parser.mutable_label_cpu(d) + nnum_cum_layer[p];
        const float* src = octree_parsers[i].label_cpu(d);
        for (int j = 0; j < nnum[p]; ++j) { des[j] = src[j]; }
      }

      // copy split label: !NOTE! the type of label is float!!!
      int split_location = info_batch.locations(OctreeInfo::kSplit);
      depth_start = split_location == depth ? depth : 0;
      for (int d = depth_start; d < depth + 1; ++d) {
        if (!info_batch.has_property(OctreeInfo::kSplit)) break;
        int p = i * (depth + 1) + d;
        float* des = octbatch_parser.mutable_split_cpu(d) + nnum_cum_layer[p];
        const float* src = octree_parsers[i].split_cpu(d);
        for (int j = 0; j < nnum[p]; ++j) des[j] = src[j];
      }
    }
  };

  int thread_num = 8;
#ifdef _DEBUG
  thread_num = 1;   // for debug only
#endif
  vector<shared_ptr<boost::thread> > workers(thread_num);
  for (int id = 1; id < thread_num; ++id) {
    workers[id].reset(new boost::thread(worker, id, thread_num));
  }
  worker(0, thread_num); // for the master thread
  for (int id = 1; id < thread_num; ++id) {
    workers[id]->join();
  }

  // ==== v2 ====
  // calc and set neighbor info
  for (int d = 1; d < depth + 1; ++d) {
    if (!info_batch.has_property(OctreeInfo::kNeigh)) break;
    CHECK(info_batch.has_property(OctreeInfo::kChild));

    if (d <= full_layer) {
      octree::calc_neigh_cpu(
          octbatch_parser.mutable_neighbor_cpu(d),
          d, batch_size);
    } else {
      octree::calc_neigh_cpu(
          octbatch_parser.mutable_neighbor_cpu(d),
          octbatch_parser.neighbor_cpu(d - 1),
          octbatch_parser.children_cpu(d - 1),
          octbatch_parser.info().node_num(d - 1));
    }
  }
}

template<typename Dtype>
void set_octree_parser(OctreeParser& octree_parser, const Blob<Dtype>& octree_in) {
  if (Caffe::mode() == Caffe::CPU) {
    octree_parser.set_cpu(octree_in.cpu_data());
  } else {
    const Dtype* ptr_gpu = octree_in.gpu_data();
    // After calling gpu_data(), the data head equals HEAD_AT_GPU or SYNCED
    if (octree_in.data()->head() == SyncedMemory::HEAD_AT_GPU) {
      octree_parser.set_gpu(ptr_gpu);
    } else {
      // If the data head is SYNCED, just reuse the pointer on cpu by calling
      // cpu_data() to avoid copying data from gpu to cpu
      octree_parser.set_gpu(ptr_gpu, octree_in.cpu_data());
    }
  }
}

void search_key_cpu(int* idx, const int unsigned* key, const int n_key,
    const unsigned int* query, const int n_query) {
  for (int i = 0; i < n_query; ++i) {
    int j = std::lower_bound(key, key + n_key, query[i]) - key;
    idx[i] = (j >= n_key || key[j] != query[i]) ? -1 : j;
  }
}

int content_flag(string str) {
  // The tokens are in correspondence with OctreeInfo::PropType
  const vector<string> tokens{
    "key", "child", "neigh", "feature", "label", "split"
  };

  int flag = 0;
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  for (int i = 0; i < tokens.size(); ++i) {
    size_t pos = str.find(tokens[i]);
    if (string::npos != pos) {
      flag |= (1 << i);
    }
  }
  return flag;
}

// Explicit instantiation
template void pad_forward_cpu<float>(float* Y, const int Hy,
    const int Cy, const float* X, const int Hx, const int* label);
template void pad_forward_cpu<double>(double* Y, const int Hy,
    const int Cy, const double* X, const int Hx, const int* label);
template void pad_backward_cpu<float>(float* X, const int Hx,
    const int Cx, const float* Y, const int Hy, const int* label);
template void pad_backward_cpu<double>(double* X, const int Hx,
    const int Cx, const double* Y, const int Hy, const int* label);
template void octree2col_cpu<float>(float* data_col,
    const float* data_octree, const int channel, const int height,
    const int kernel_sdim, const int stride, const int* neigh,
    const int* ni, const int height_col, const int n);
template void octree2col_cpu<double>(double* data_col,
    const double* data_octree, const int channel, const int height,
    const int kernel_sdim, const int stride, const int* neigh,
    const int* ni, const int height_col, const int n);
template void col2octree_cpu<float>(const float* data_col,
    float* data_octree, const int channel, const int height,
    const int kernel_sdim, const int stride, const int* neigh,
    const int* ni, const int height_col, const int n);
template void col2octree_cpu<double>(const double* data_col,
    double* data_octree, const int channel, const int height,
    const int kernel_sdim, const int stride, const int* neigh,
    const int* ni, const int height_col, const int n);
template void generate_label_cpu<float>(int* label_data, int& top_h,
    const float* bottom_data, const int bottom_h, const int mask);
template void generate_label_cpu<double>(int* label_data, int& top_h,
    const double* bottom_data, const int bottom_h, const int mask);
template void merge_octrees<float>(Blob<float>& octree_output,
    const vector<vector<char> >& octrees);
template void merge_octrees<double>(Blob<double>& octree_output,
    const vector<vector<char> >& octrees);
template void set_octree_parser<float>(OctreeParser& octree_parser,
    const Blob<float>& octree_in);
template void set_octree_parser<double>(OctreeParser& octree_parser,
    const Blob<double>& octree_in);

}  // namespace octree

}  // namespace caffe