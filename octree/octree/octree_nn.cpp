#include "logs.h"
#include "octree_nn.h"
#include <cstring>
#include <algorithm>

void NeighHelper::init_neigh_index() {
  const vector<std::pair<string, int> > kernel_type{
    { "333", 0 }, { "111", 1 }, { "222", 2 },
    { "311", 3 }, { "131", 4 }, { "113", 5 },
    { "331", 6 }, { "313", 7 }, { "133", 8 } };

  const vector<vector<int> > vec{ {} /* 333, 27 */, { 13 } /* 111, 1 */,
    { 13, 14, 16, 17, 22, 23, 25, 26 } /* 222, 8, 8 octants */,
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

  // init the bilinear table
  bilinear_.assign(512, -1);
  const int mask[8][3] = {                       // bilinear weights:
    {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},  // 27, 9, 9, 9
    {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},  //  3, 3, 3, 1
  };
  for (int i = 0; i < 8; ++i) {
    // i -> xyz
    int z0 = i % 2, t = i / 2;
    int y0 = t % 2, x0 = t / 2;

    for (int j = 0; j < 8; ++j) {
      // j -> xyz
      int z1 = j % 2, s = j / 2;
      int y1 = s % 2, x1 = s / 2;

      for (int k = 0; k < 8; ++k) {
        int x2 = x0 + 1 + mask[k][0] * (2 * x1 - 1);
        int y2 = y0 + 1 + mask[k][1] * (2 * y1 - 1);
        int z2 = z0 + 1 + mask[k][2] * (2 * z1 - 1);

        bilinear_[(i << 6) | (j << 3) | k] = (x2 << 4) | (y2 << 2) | z2;
      }
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


int num_elements(const vector<int>& vec) {
  int count = vec.empty() ? 0 : 1;
  for (auto v : vec) { count *= v; }
  return count;
}

template<typename Dtype>
void resize_with_last_val(vector<Dtype>& vec, const int size) {
  int len = vec.size();
  if (len == 0) return;
  if (len < size) {
    int v = vec[len - 1];
    for (int i = len; i < size; ++i) {
      vec.push_back(v);
    }
  } else {
    vec.resize(size);
  }
}


template <typename Dtype>
void memset_cpu(const size_t N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);
    return;
  }
  for (size_t i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}


template <typename Dtype>
void memcpy_cpu(const size_t N, const Dtype* X, Dtype* Y) {
  if (X != Y && N > 0) {
    memcpy(Y, X, sizeof(Dtype) * N);
  }
}


template<typename Dtype>
void pad_forward_cpu(Dtype* Y, const int Hy,
    const int Cy, const Dtype* X, const int Hx, const int* label, const Dtype dval) {
  // Note: Cx == Cy
  for (int c = 0; c < Cy; ++c) {
    for (int h = 0; h < Hy; ++h) {
      Y[c * Hy + h] = label[h] == -1 ? dval : X[c * Hx + label[h]];
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


template <typename Dtype>
void octree2colP_cpu(Dtype* data_col, const Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n) {
  for (int c = 0; c < channel; ++c) {
    for (int k = 0; k < kernel_sdim; ++k) {
      int h_start = n * height_col;
      int i_start = (c * kernel_sdim + k) * height_col - h_start;
      for (int h = h_start; h < h_start + height_col; ++h) {
        // boundary condition
        if (h >= height) {
          data_col[i_start + h] = Dtype(0);
          continue;
        }
        // neighborhood searching
        const int hp = ichild[h];
        const int index = stride == 2 ? (h << 6) + ni[k] :
            (hp >> 3 << 6) + ni[(hp % 8) * kernel_sdim + k];
        int p = neigh[index];
        if (p >= 0) { p = child[p]; }
        // assign values
        data_col[i_start + h] =
            p < 0 ? Dtype(0) : data_octree[c * octree_h + p];
      }
    }
  }
}

template <typename Dtype>
void col2octreeP_cpu(const Dtype* data_col, Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n) {
  // set data_octree to zero ONCE when n ==0
  if (n == 0) { memset_cpu(channel * octree_h, Dtype(0), data_octree); }
  for (int c = 0; c < channel; ++c) {
    for (int k = 0; k < kernel_sdim; ++k) {
      int h_start = n * height_col;
      int i_start = (c * kernel_sdim + k) * height_col - h_start;
      for (int h = h_start; h < h_start + height_col; ++h) {
        // boundary condition
        if (h >= height) continue;
        // neighborhood searching
        const int hp = ichild[h];
        const int index = stride == 2 ? (h << 6) + ni[k] :
            (hp >> 3 << 6) + ni[(hp % 8) * kernel_sdim + k];
        int p = neigh[index];
        if (p >= 0) { p = child[p]; }
        // assign values
        if (p >= 0) { data_octree[c * octree_h + p] += data_col[i_start + h]; }          
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
  const uintk bit = 1;
  uintk node_num = bit << 3 * depth;
  const uintk  bound = 1 << depth;
  for (uintk n = 0; n < batch_size; ++n) {
    for (uintk i = 0; i < node_num; i += 8) {
      // key to xyz
      uintk x0 = 0, y0 = 0, z0 = 0;
      for (uintk d = 0; d < depth; d++) {
        x0 |= (i & (bit << (3 * d + 2))) >> (2 * d + 2);
        y0 |= (i & (bit << (3 * d + 1))) >> (2 * d + 1);
        z0 |= (i & (bit << (3 * d + 0))) >> (2 * d + 0);
      }

      for (uintk x = 0; x < 4; ++x) {
        for (uintk y = 0; y < 4; ++y) {
          for (uintk z = 0; z < 4; ++z) {
            uintk x1 = x0 + x - 1;
            uintk y1 = y0 + y - 1;
            uintk z1 = z0 + z - 1;

            int v = -1;
            if ((x1 & bound) == 0 &&
                (y1 & bound) == 0 &&
                (z1 & bound) == 0) {
              uintk key1 = 0;
              for (int d = 0; d < depth; d++) {
                uintk mask = 1u << d;
                key1 |= ((x1 & mask) << (2 * d + 2)) |
                    ((y1 & mask) << (2 * d + 1)) |
                    ((z1 & mask) << (2 * d));
              }
              v = key1 + n * node_num;
            }

            uintk xyz = (x << 4) | (y << 2) | z;
            neigh[xyz + i * 8 + n * node_num * 8] = v;
          }
        }
      }
    }
  }
}


template <typename Dtype>
void generate_key_cpu(Dtype* key_child, const Dtype* key, const int* child,
    const int node_num) {
  typedef typename KeyTrait<Dtype>::uints T;
  for (int i = 0; i < node_num; ++i) {
    int label = child[i];
    if (label < 0) continue;  // empty
    const T* k0 = reinterpret_cast<const T*>(key + i);
    for (T j = 0; j < 8; ++j) {
      T* k1 = reinterpret_cast<T*>(key_child + 8 * label + j);
      k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
      k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
      k1[2] = (k0[2] << 1) | (j & 1);
      k1[3] =  k0[3];
    }
  }
}

template <typename Dtype>
void generate_key_cpu(Dtype* key, const int depth, const int batch_size) {
  typedef typename KeyTrait<Dtype>::uints T;
  const Dtype bit = 1;
  int node_num = bit << 3 * depth;
  for (int n = 0; n < batch_size; ++n) {
    for (int k = 0; k < node_num; ++k) {
      Dtype xyz = 0;
      T* ptr = reinterpret_cast<T*>(&xyz);
      for (int d = 0; d < depth; d++) {
        ptr[0] |= (k & (bit << (3 * d + 2))) >> (2 * d + 2);
        ptr[1] |= (k & (bit << (3 * d + 1))) >> (2 * d + 1);
        ptr[2] |= (k & (bit << (3 * d + 0))) >> (2 * d + 0);
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


void bilinear_neigh_cpu(int* bidx, const int* neigh, const int* child,
    const int node_num, const int* table) {
  for (int i = 0; i < node_num; ++i) {
    int cld = child[i];
    if (cld < 0) continue;    // skip empty node

    const int* nghi = neigh + (i >> 3 << 6);
    for (int j = 0; j < 8; ++j) {
      int k = (cld * 8 + j);  // child id
      int* des = bidx + k * 8;
      const int* tb = table + ((i % 8) * 8 + j) * 8;
      for (int k = 0; k < 8; ++k) {
        des[k] = nghi[tb[k]];
      }
    }
  }
}

void bilinear_xyz_cpu(uintk* xyz0, float* fracs, const int d0, const uintk* xyz1,
    const int d1, const int num) {
  typedef typename KeyTrait<uintk>::uints uints;
  const float scale = static_cast<float>(1 << (d1 - d0));
  const int mask[8][3] = {                       // bilinear mask:
    {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},  // 27, 9, 9, 9
    {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},  //  3, 3, 3, 1
  };

  for (int i = 0; i < num; ++i) {
    float pt[3] = { 0.0f };
    float* frac = fracs + 3 * i;
    int bnd[2][3] = { 0 };
    const uints* ptr1 = (const uints*)(xyz1 + i);
    for (int c = 0; c < 3; ++c) {
      pt[c] = (static_cast<float>(ptr1[c]) + 0.5f) / scale - 0.5f;

      int b = static_cast<int>(pt[c]);
      frac[c] = pt[c] - static_cast<float>(b);
      if (frac[c] > 0.5f) {
        bnd[0][c] = b + 1;
        bnd[1][c] = b;
      } else {
        frac[c] = 1 - frac[c];
        bnd[0][c] = b;
        bnd[1][c] = b + 1;
      }
    }

    for (int j = 0; j < 8; ++j) {
      uints* ptr0 = (uints*)(xyz0 + i * 8 + j);
      for (int c = 0; c < 3; ++c) {
        ptr0[c] = static_cast<uints>(bnd[mask[j][c]][c]);
      }
      ptr0[3] = ptr1[3];
    }
  }
}


template <typename Dtype>
void sequence_cpu(Dtype* ptr, const int num) {
  for (int i = 0; i < num; ++i) {
    ptr[i] = static_cast<Dtype>(i);
  }
}

template <typename Dtype>
void search_key_cpu(int* idx, const Dtype* key, const int n_key,
    const Dtype* query, const int n_query) {
  for (int i = 0; i < n_query; ++i) {
    int j = std::lower_bound(key, key + n_key, query[i]) - key;
    idx[i] = (j >= n_key || key[j] != query[i]) ? -1 : j;
  }
}


template<typename Dtype>
void compute_key(Dtype& key, const Dtype* pt, const int depth) {
  key = 0;
  for (int i = 0; i < depth; i++) {
    Dtype mask = 1u << i;
    for (int j = 0; j < 3; j++) {
      key |= (pt[j] & mask) << (2 * i + 2 - j);
    }
  }
}


template<typename Dtype>
void compute_pt(Dtype* pt, const Dtype& key, const int depth) {
  for (int i = 0; i < 3; pt[i++] = 0u);
  const Dtype bit = 1;
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < 3; j++) {
      // bit mask
      Dtype mask = bit << (3 * i + 2 - j);
      // put the bit to position i
      pt[j] |= (key & mask) >> (2 * i + 2 - j);
    }
  }
}


template<typename Dtype>
void xyz2key_cpu(Dtype* key, const Dtype* xyz, const int num, const int depth) {
  typedef typename KeyTrait<Dtype>::uints T;

  for (int i = 0; i < num; ++i) {
    Dtype pt[3] = { 0, 0, 0 }, key_out = 0;
    const T* ptr = reinterpret_cast<const T*>(xyz + i);
    T* ptr_out = reinterpret_cast<T*>(&key_out);
    for (int j = 0; j < 3; ++j) {
      pt[j] = static_cast<Dtype>(ptr[j]);
    }
    compute_key(key_out, pt, depth);
    ptr_out[3] = ptr[3];
    key[i] = key_out;
  }
}

template<typename Dtype>
void key2xyz_cpu(Dtype* xyz, const Dtype* key, const int num, const int depth) {
  typedef typename KeyTrait<Dtype>::uints T;

  for (int i = 0; i < num; ++i) {
    Dtype pt[3] = { 0 };
    compute_pt(pt, key[i], depth);

    xyz[i] = key[i];
    T* ptr = reinterpret_cast<T*>(xyz + i);
    for (int j = 0; j < 3; ++j) {
      ptr[j] = static_cast<T>(pt[j]);
    }
  }
}


template<typename Dtype>
void key2idx_cpu(int* idx, const Dtype* key, const int num) {
  typedef typename KeyTrait<Dtype>::uints T;
  for (int i = 0; i < num; ++i) {
    const T* ptr = reinterpret_cast<const T*>(key + i);
    idx[i] = static_cast<int>(ptr[3]);
  }
}


template<typename Dtype>
void xyz2coord_cpu(float* pt, const Dtype* xyz, const int num, const int channel) {
  typedef typename KeyTrait<Dtype>::uints T;
  for (int i = 0; i < num; ++i) {
    const T* ptr = reinterpret_cast<const T*>(xyz + i);
    for (int c = 0; c < channel; ++c) {
      pt[c * num + i] = static_cast<float>(ptr[c]);
    }
  }
}


template<typename Dtype>
void coord2xyz_cpu(Dtype* xyz, const float* pt, const int num, const int channel) {
  typedef typename KeyTrait<Dtype>::uints T;
  for (int i = 0; i < num; ++i) {
    T* ptr = reinterpret_cast<T*>(xyz + i);
    for (int c = 0; c < channel; ++c) {
      ptr[c] = static_cast<T>(pt[c * num + i]);
    }
  }
}


template<typename Dtype1, typename Dtype2>
void key2xyz(Dtype1* xyz, const Dtype2 key, const int depth) {
  Dtype2 pt[3];
  compute_pt(pt, key, depth);
  for (int c = 0; c < 3; ++c) {
    xyz[c] = static_cast<Dtype1>(pt[c]);
  }
}

// Explicit instantiation
template void resize_with_last_val<int>(vector<int>& vec, const int sz);
template void resize_with_last_val<float>(vector<float>& vec, const int size);
template void memset_cpu<int>(const size_t N, const int alpha, int* Y);
template void memset_cpu<float>(const size_t N, const float alpha, float* Y);
template void memset_cpu<double>(const size_t N, const double alpha, double* Y);
template void memset_cpu<char>(const size_t N, const char alpha, char* Y);
template void memset_cpu<int8_t>(const size_t N, const int8_t alpha, int8_t* Y);
template void memset_cpu<uint8_t>(const size_t N, const uint8_t alpha, uint8_t* Y);
template void memcpy_cpu<int>(const size_t N, const int* X, int* Y);
template void memcpy_cpu<uint32>(const size_t N, const uint32* X, uint32* Y);
template void memcpy_cpu<uint64>(const size_t N, const uint64* X, uint64* Y);
template void memcpy_cpu<float>(const size_t N, const float* X, float* Y);
template void memcpy_cpu<double>(const size_t N, const double* X, double* Y);
template void sequence_cpu<int>(int* ptr, const int num);
template void sequence_cpu<uintk>(uintk* ptr, const int num);
template void pad_forward_cpu<float>(float* Y, const int Hy, const int Cy,
    const float* X, const int Hx, const int* label, const float dval);
template void pad_forward_cpu<double>(double* Y, const int Hy, const int Cy,
    const double* X, const int Hx, const int* label, const double dval);
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
template void octree2colP_cpu<float>(float* data_col, const float* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
template void col2octreeP_cpu<float>(const float* data_col, float* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
template void octree2colP_cpu<double>(double* data_col, const double* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
template void col2octreeP_cpu<double>(const double* data_col, double* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
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
template void key2xyz<float, uintk>(float* xyz, const uintk key, const int d);
template void key2xyz<uintk, uintk>(uintk* xyz, const uintk key, const int d);
template void key2xyz<int, uintk>(int* xyz, const uintk key, const int d);
template void generate_key_cpu<uintk>(uintk* key, const int depth, const int batch_size);
template void generate_key_cpu<uintk>(uintk* key_child, const uintk* key,
    const int* child, const int node_num);
template void compute_key<uintk>(uintk& key, const uintk* pt, const int depth);
template void compute_pt<uintk>(uintk* pt, const uintk& key, const int depth);
template void search_key_cpu<uintk>(int* idx, const uintk* key, const int n_key,
    const uintk* query, const int n_query);
template void xyz2key_cpu<uintk>(uintk* key, const uintk* xyz, const int num,
    const int depth);
template void key2xyz_cpu<uintk>(uintk* xyz, const uintk* key, const int num,
    const int depth);
template void key2idx_cpu<uintk>(int* idx, const uintk* key, const int num);
template void xyz2coord_cpu<uintk>(float* pt, const uintk* xyz, const int num,
    const int channel);
template void coord2xyz_cpu<uintk>(uintk* xyz, const float* pt, const int num,
    const int channel);