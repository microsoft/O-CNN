#include "logs.h"
#include "octree_util.h"
#include <cstring>
#include <algorithm>

void NeighHelper::init_neigh_index() {
  const vector<std::pair<string, int> > kernel_type{
    { "333", 0 }, { "111", 1 }, { "222", 2 },
    { "311", 3 }, { "131", 4 }, { "113", 5 },
    { "331", 6 }, { "313", 7 }, { "133", 8 } };

  const vector<vector<int> > vec{ {} /* 333, 27 */, { 13 } /* 111, 1 */,
    { 13, 14, 16, 17, 22, 23, 25, 26 } /* 222, 8 */,
    {  4, 13, 22 } /* 311, 3 */,
    { 10, 13, 16 } /* 131, 3 */,
    { 12, 13, 14 } /* 113, 3 */,
    { 1,  4,  7, 10, 13, 16, 19, 22, 25 } /* 331, 9 */,
    { 3,  4,  5, 12, 13, 14, 21, 22, 23 } /* 313, 9 */,
    { 9, 10, 11, 12, 13, 14, 15, 16, 17 } /* 133, 9 */ };

  // init
  ni_map_.insert(kernel_type.begin(), kernel_type.end());
  ni_.resize(kernel_type.size());

  // ni for kernel_size=333
  ni_[0].assign(216, 0);
  int* ni3 = ni_[0].data();
  int id = 0;
  for (int ijk = 0; ijk < 8; ++ijk) {
    for (int xyz = 0; xyz < 27; ++xyz) {
      int k = ijk % 2, p = ijk / 2;
      int j = p % 2, i = p / 2;

      int z = xyz % 3, q = xyz / 3;
      int y = q % 3, x = q / 3;

      ni3[id++] = ((x + i) << 4) | ((y + j) << 2) | (z + k);
    }
  }

  // ni for other kernel_sizes
  for (int k = 1; k < kernel_type.size(); ++k) {
    int sz = vec[k].size();
    ni_[k].assign(8 * sz, 0);
    int* ni = ni_[k].data();
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < sz; ++j) {
        ni[i * sz + j] = ni3[i * 27 + vec[k][j]];
      }
    }
  }

  // init the array parent & displacement
  id = 0;
  int tmp[64];
  displacement_.assign(64, 0);
  int* dis_ptr = displacement_.data();
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

  parent_.assign(512, 0);
  int* parent_ptr = parent_.data();
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 64; ++j) {
      parent_ptr[i * 64 + j] = ni3[i * 27 + tmp[j]];
    }
  }
}

vector<int>& NeighHelper::get_ni(const vector<int>& kernel_size) {
  string key;
  CHECK(kernel_size.size() == 3);
  for (auto i : kernel_size) key += std::to_string(i);

  auto it = Get().ni_map_.find(key);
  CHECK(!(it == Get().ni_map_.end())) << "Unsupported kernel_size";
  return Get().ni_[it->second];
}


template <typename Dtype>
void memset_cpu(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

int num_elements(const vector<int>& vec) {
  int count = vec.empty() ? 0 : 1;
  for (auto v : vec) { count *= v; }
  return count;
}

template <typename Dtype>
void memcpy_cpu(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * N);
  }
}


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
  if (n == 0) memset_cpu(channel * octree_h, Dtype(0), data_octree);
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

template<typename Dtype>
void octree_max_pool_cpu(Dtype* top_data, int top_h, int* mask,
    const Dtype* btm_data, int btm_h, int channel) {
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < top_h; ++h) {
      int hb = 8 * h;
      top_data[h] = btm_data[hb];
      mask[h] = hb;
      for (int idx = hb + 1; idx < hb + 8; ++idx) {
        if (btm_data[idx] > top_data[h]) {
          top_data[h] = btm_data[idx];
          mask[h] = idx;
        }
      }
    }

    // update pointer
    mask += top_h;
    top_data += top_h;
    btm_data += btm_h;
  }
}

template<typename Dtype>
void octree_max_unpool_cpu(const Dtype* top_data, int top_h, const int* mask,
    Dtype* btm_data, int btm_h, int channel) {
  memset_cpu(btm_h * channel, Dtype(0), btm_data);

  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < top_h; ++h) {
      btm_data[mask[h]] = top_data[h];
    }

    // update pointer
    mask += top_h;
    top_data += top_h;
    btm_data += btm_h;
  }
}

template<typename Dtype>
void octree_mask_pool_cpu(Dtype* top_data, int top_h, const int* mask,
    const Dtype* btm_data, int btm_h, int channel) {
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < top_h; ++h) {
      top_data[h] = btm_data[mask[h]];
    }

    // update pointer
    mask += top_h;
    top_data += top_h;
    btm_data += btm_h;
  }
}


void calc_neigh_cpu(int* neigh_split, const int* neigh,
    const int* children, const int node_num) {
  const int* parent = NeighHelper::Get().get_parent_array().data();
  const int* dis = NeighHelper::Get().get_dis_array().data();

  //#pragma omp parallel for
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


void generate_key_cpu(uint32* key_child, const uint32* key, const int* child,
    const int node_num) {
  typedef unsigned char ubyte;
  for (int i = 0; i < node_num; ++i) {
    int label = child[i];
    if (label == -1) continue;
    const ubyte* k0 = (const ubyte*)(key + i);
    for (ubyte j = 0; j < 8; ++j) {
      ubyte* k1 = (ubyte*)(key_child + 8 * label + j);
      k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
      k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
      k1[2] = (k0[2] << 1) | (j & 1);
      k1[3] =  k0[3];
    }
  }
}

void generate_key_cpu(uint32* key, const int depth, const int batch_size) {
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


template <typename Dtype>
void generate_label_cpu(int* label_data, int& top_h, const Dtype* bottom_data,
    const int bottom_h, const int mask) {
  top_h = 0;
  for (int i = 0; i < bottom_h; ++i) {
    label_data[i] = (mask == static_cast<int>(bottom_data[i])) ? top_h++ : -1;
  }
}


void search_key_cpu(int* idx, const uint32* key, const int n_key,
    const uint32* query, const int n_query) {
  for (int i = 0; i < n_query; ++i) {
    int j = std::lower_bound(key, key + n_key, query[i]) - key;
    idx[i] = (j >= n_key || key[j] != query[i]) ? -1 : j;
  }
}


void compute_key(uint32& key, const uint32* pt, const int depth) {
  key = 0;
  for (int i = 0; i < depth; i++) {
    uint32 mask = 1u << i;
    for (int j = 0; j < 3; j++) {
      key |= (pt[j] & mask) << (2 * i + 2 - j);
    }
  }
}

void compute_pt(uint32* pt, const uint32& key, const int depth) {
  for (int i = 0; i < 3; pt[i++] = 0u);

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < 3; j++) {
      // bit mask
      uint32 mask = 1u << (3 * i + 2 - j);
      // put the bit to position i
      pt[j] |= (key & mask) >> (2 * i + 2 - j);
    }
  }
}


// NOTE: !!! currently the depth should be less than 8
void xyz2key_cpu(uint32* key, const uint32* xyz, const int num, const int depth) {
  for (int i = 0; i < num; ++i) {
    uint32 pt[3] = { 0, 0, 0 }, key_out = 0;
    const unsigned char* ptr = reinterpret_cast<const unsigned char*>(xyz + i);
    unsigned char* ptr_out = (unsigned char*)(&key_out);
    for (int j = 0; j < 3; ++j) {
      pt[j] = static_cast<uint32>(ptr[j]);
    }
    compute_key(key_out, pt, depth);
    ptr_out[3] = ptr[3];
    key[i] = key_out;
  }
}

void key2xyz_cpu(uint32* xyz, const uint32* key, const int num, const int depth) {
  for (int i = 0; i < num; ++i) {
    uint32 pt[3] = { 0 };
    compute_pt(pt, key[i], depth);

    xyz[i] = key[i];
    unsigned char* ptr = reinterpret_cast<unsigned char*>(xyz + i);
    for (int j = 0; j < 3; ++j) {
      ptr[j] = static_cast<unsigned char>(pt[j]);
    }
  }
}


template<typename Dtype>
void key2xyz(Dtype* xyz, const uint32 key, const int depth) {
  uint32 pt[3];
  compute_pt(pt, key, depth);
  for (int c = 0; c < 3; ++c) {
    xyz[c] = static_cast<Dtype>(pt[c]);
  }
}

// Explicit instantiation
template void memset_cpu<int>(const int N, const int alpha, int* Y);
template void memset_cpu<float>(const int N, const float alpha, float* Y);
template void memset_cpu<double>(const int N, const double alpha, double* Y);
template void memset_cpu<char>(const int N, const char alpha, char* Y);
template void memset_cpu<int8_t>(const int N, const int8_t alpha, int8_t* Y);
template void memset_cpu<uint8_t>(const int N, const uint8_t alpha, uint8_t* Y);
template void memcpy_cpu<int>(const int N, const int* X, int* Y);
template void memcpy_cpu<unsigned>(const int N, const unsigned* X, unsigned* Y);
template void memcpy_cpu<float>(const int N, const float* X, float* Y);
template void memcpy_cpu<double>(const int N, const double* X, double* Y);
template void pad_forward_cpu<float>(float* Y, const int Hy, const int Cy,
    const float* X, const int Hx, const int* label);
template void pad_forward_cpu<double>(double* Y, const int Hy, const int Cy,
    const double* X, const int Hx, const int* label);
template void pad_backward_cpu<float>(float* X, const int Hx, const int Cx,
    const float* Y, const int Hy, const int* label);
template void pad_backward_cpu<double>(double* X, const int Hx, const int Cx,
    const double* Y, const int Hy, const int* label);
template void octree2col_cpu<float>(float* data_col, const float* data_octree,
    const int channel, const int height,  const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void octree2col_cpu<double>(double* data_col, const double* data_octree,
    const int channel, const int height, const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void col2octree_cpu<float>(const float* data_col, float* data_octree,
    const int channel, const int height, const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void col2octree_cpu<double>(const double* data_col, double* data_octree,
    const int channel, const int height, const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void generate_label_cpu<float>(int* label_data, int& top_h,
    const float* bottom_data, const int bottom_h, const int mask);
template void generate_label_cpu<double>(int* label_data, int& top_h,
    const double* bottom_data, const int bottom_h, const int mask);
template void generate_label_cpu<int>(int* label_data, int& top_h,
    const int* bottom_data, const int bottom_h, const int mask);
template void octree_max_pool_cpu<float>(float* top_data, int top_h,
    int* mask, const float* btm_data, int bottom_h, int channel);
template void octree_max_pool_cpu<double>(double* top_data, int top_h,
    int* mask, const double* btm_data, int bottom_h, int channel);
template void octree_max_unpool_cpu<float>(const float* top_data, int top_h,
    const int* mask, float* btm_data, int bottom_h, int channel);
template void octree_max_unpool_cpu<double>(const double* top_data, int top_h,
    const int* mask, double* btm_data, int bottom_h, int channel);
template void octree_mask_pool_cpu<float>(float* top_data, int top_h,
    const int* mask, const float* btm_data, int bottom_h, int channel);
template void octree_mask_pool_cpu<double>(double* top_data, int top_h,
    const int* mask, const double* btm_data, int bottom_h, int channel);
template void key2xyz<float>(float* xyz, const uint32 key, const int d);
template void key2xyz<uint32>(unsigned* xyz, const uint32 key, const int d);
template void key2xyz<int>(int* xyz, const uint32 key, const int d);