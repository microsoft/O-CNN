#include "octree_nn.h"
#include "device_alternate.h"

#include <climits>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>


template <typename Dtype>
inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

template <>
inline __device__ float caffe_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__ double caffe_gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}


template <typename Dtype>
__global__ void memset_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void memset_gpu(const size_t N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));
    return;
  }
  CHECK(N < INT_MAX) << "Overflow in memset_gpu";
  memset_kernel<Dtype> <<< CudaGetBlocks(N), kCudaThreadsNum >>> (
      N, alpha, Y);
}

template <typename Dtype>
void memcpy_gpu(const size_t N, const Dtype* X, Dtype* Y) {
  if (X != Y && N > 0) {
    CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
  }
}


template <typename Dtype>
__global__ void pad_forward_kernel(Dtype* Y, const int Hy,
    const Dtype* X, const int Hx, const int* label, const int n, const Dtype dval) {
  CUDA_KERNEL_LOOP(i, n) {
    int h = i % Hy;
    int c = i / Hy;

    int idx = label[h];
    Y[i] = idx == -1 ? dval : X[c * Hx + idx];
  }
}

template <typename Dtype>
__global__ void pad_backward_kernel(Dtype* X, const int Hx,
    const Dtype* Y, const int Hy, const int* label, const int n) {
  CUDA_KERNEL_LOOP(i, n) {
    int h = i % Hy;
    int c = i / Hy;

    int idx = label[h];
    if (idx != -1) {
      X[c * Hx + idx] = Y[i];
    }
  }
}

template<typename Dtype>
void pad_forward_gpu(Dtype* Y, const int Hy, const int Cy,
    const Dtype* X, const int Hx, const int* label, const Dtype dval) {
  int n = Hy * Cy; // Note: Cx == Cy
  pad_forward_kernel<Dtype> <<< CudaGetBlocks(n), kCudaThreadsNum >>> (
      Y, Hy, X, Hx, label, n, dval);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void pad_backward_gpu(Dtype* X, const int Hx, const int Cx,
    const Dtype* Y, const int Hy, const int* label) {
  int n = Hy * Cx; // Note: Cx == Cy
  pad_backward_kernel<Dtype> <<< CudaGetBlocks(n), kCudaThreadsNum >>> (
      X, Hx, Y, Hy, label, n);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void octree2col_kernel(Dtype* data_col, const Dtype* data_octree,
    const int height, const int kernel_dim, const int stride, const int* neigh,
    const int* ni, const int height_col, const int n, const int thread_num) {
  CUDA_KERNEL_LOOP(i, thread_num) {
    int h = i % height_col;
    int h1 = h + n * height_col;
    if (h1 >= height) { data_col[i] = 0; continue; }
    int t = i / height_col;
    int k = t % kernel_dim;
    int c = t / kernel_dim;
    int octree_h = height << 3 * (stride - 1);

    int index = stride == 2 ? (h1 << 6) + ni[k] :
        (h1 >> 3 << 6) + ni[(h1 % 8) * kernel_dim + k];
    int p = neigh[index];
    data_col[i] = p == -1 ? Dtype(0) : data_octree[c * octree_h + p];
  }
}

template <typename Dtype>
__global__ void col2octree_kernel(const Dtype* data_col, Dtype* data_octree,
    const int height, const int kernel_dim, const int stride, const int* neigh,
    const int* ni, const int height_col, const int n, const int thread_num) {
  CUDA_KERNEL_LOOP(i, thread_num) {
    int h = i % height_col;
    int h1 = h + n * height_col;
    if (h1 >= height) continue;
    int t = i / height_col;
    int k = t % kernel_dim;
    int c = t / kernel_dim;
    int octree_h = height << 3 * (stride - 1);

    int index = stride == 2 ? (h1 << 6) + ni[k] :
        (h1 >> 3 << 6) + ni[(h1 % 8) * kernel_dim + k];
    int p = neigh[index];
    if (p != -1) caffe_gpu_atomic_add(data_col[i], data_octree + c * octree_h + p);
  }
}

template <typename Dtype>
void octree2col_gpu(Dtype* data_col, const Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n) {
  const int thread_num = channel * kernel_sdim * height_col;
  octree2col_kernel<Dtype> <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      data_col, data_octree, height, kernel_sdim, stride, neigh, ni, height_col,
      n, thread_num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void col2octree_gpu(const Dtype* data_col, Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n) {
  const int thread_num = channel * kernel_sdim * height_col;
  int octree_h = height << 3 * (stride - 1);
  // set data_octree to zero ONCE when n ==0
  if (n == 0) memset_gpu(channel * octree_h, Dtype(0), data_octree);
  col2octree_kernel<Dtype> <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      data_col, data_octree, height, kernel_sdim, stride, neigh, ni, height_col,
      n, thread_num);
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
__global__ void octree2colP_kernel(Dtype* data_col, const Dtype* data_octree,
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild,
    const int height_col, const int n, const int thread_num) {
  CUDA_KERNEL_LOOP(i, thread_num) {
    int h = i % height_col;
    int h1 = h + n * height_col;
    if (h1 >= height) { data_col[i] = 0; continue; }
    int t = i / height_col;
    int k = t % kernel_sdim;
    int c = t / kernel_sdim;
    
    // neighborhood searching
    const int hp = ichild[h];
    const int index = stride == 2 ? (h << 6) + ni[k] :
        (hp >> 3 << 6) + ni[(hp % 8) * kernel_sdim + k];
    int p = neigh[index];
    if (p >= 0) { p = child[p]; }

    data_col[i] = p < 0 ? Dtype(0) : data_octree[c * octree_h + p];
  }
}

template <typename Dtype>
__global__ void col2octreeP_kernel(const Dtype* data_col, Dtype* data_octree,
    const int height, const int octree_h, const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int* child, const int* ichild,
    const int height_col, const int n, const int thread_num) {
  CUDA_KERNEL_LOOP(i, thread_num) {
    int h = i % height_col;
    int h1 = h + n * height_col;
    if (h1 >= height) continue;
    int t = i / height_col;
    int k = t % kernel_sdim;
    int c = t / kernel_sdim;

    // neighborhood searching
    const int hp = ichild[h];
    const int index = stride == 2 ? (h << 6) + ni[k] :
        (hp >> 3 << 6) + ni[(hp % 8) * kernel_sdim + k];
    int p = neigh[index];
    if (p >= 0) { p = child[p]; }

    // assign values
    if (p >= 0) {
      caffe_gpu_atomic_add(data_col[i], data_octree + c * octree_h + p);
    }
  }
}

template <typename Dtype>
void octree2colP_gpu(Dtype* data_col, const Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n) {
  const int thread_num = channel * kernel_sdim * height_col;
  octree2colP_kernel<Dtype> <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      data_col, data_octree, height, octree_h, kernel_sdim, stride, neigh, ni,
      child, ichild, height_col, n, thread_num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void col2octreeP_gpu(const Dtype* data_col, Dtype* data_octree, const int channel, 
    const int height, const int octree_h, const int kernel_sdim, const int stride, 
    const int* neigh, const int* ni, const int* child, const int* ichild, 
    const int height_col, const int n) {
  const int thread_num = channel * kernel_sdim * height_col;
  // set data_octree to zero ONCE when n ==0
  if (n == 0) { memset_gpu(channel * octree_h, Dtype(0), data_octree); }
  col2octreeP_kernel<Dtype> <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      data_col, data_octree, height, octree_h, kernel_sdim, stride, neigh, ni,
      child, ichild, height_col, n, thread_num);
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
__global__ void octree_max_pool_kernel(Dtype* top_data, const int top_h,
    int* mask, const Dtype* btm_data, const int btm_h, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int h = i % top_h;
    int c = i / top_h;

    int hb = 8 * h;
    int max_idx = hb;
    btm_data += c * btm_h;
    Dtype max_val = btm_data[hb];

#pragma unroll 7
    for (int idx = hb + 1; idx < hb + 8; ++idx) {
      Dtype value = btm_data[idx];
      if (value > max_val) {
        max_idx = idx;
        max_val = value;
      }
    }

    top_data[i] = max_val;
    mask[i] = max_idx;
  }
}

template<typename Dtype>
void octree_max_pool_gpu(Dtype* top_data, int top_h, int* mask,
    const Dtype* btm_data, int btm_h, int channel) {
  int num = top_h * channel;
  octree_max_pool_kernel<Dtype> <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      top_data, top_h, mask, btm_data, btm_h, num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void octree_max_unpool_kernel(const Dtype* top_data, const int top_h,
    const int* mask, Dtype* btm_data, const int btm_h,  const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int c = i / top_h;
    btm_data[c * btm_h + mask[i]] = top_data[i];
  }
}

template<typename Dtype>
void octree_max_unpool_gpu(const Dtype* top_data, int top_h, const int* mask,
    Dtype* btm_data, int btm_h, int channel) {
  int num = top_h * channel;
  memset_gpu(btm_h * channel, Dtype(0), btm_data);
  octree_max_unpool_kernel<Dtype> <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      top_data, top_h, mask, btm_data, btm_h, num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void octree_mask_pool_kernel(Dtype* top_data, const int top_h,
    const int* mask, const Dtype* btm_data, const int btm_h, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int c = i / top_h;
    top_data[i] = btm_data[c * btm_h + mask[i]];
  }
}

template<typename Dtype>
void octree_mask_pool_gpu(Dtype* top_data, int top_h, const int* mask,
    const Dtype* btm_data, int btm_h, int channel) {
  int num = top_h * channel;
  octree_mask_pool_kernel<Dtype> <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      top_data, top_h, mask, btm_data, btm_h, num);
  CUDA_POST_KERNEL_CHECK;
}



__global__ void calc_neigh_kernel(int* neigh_split, const int* neigh,
    const int* children, const int* parent, const int* dis, const int thread_num) {
  CUDA_KERNEL_LOOP(id, thread_num) {
    int i = id >> 6;
    int j = id % 64;

    int l0 = children[i];
    if (l0 != -1) {
      const int* ngh0 = neigh + (i >> 3 << 6);
      const int* pi0 = parent + (i % 8) * 64;
      int* ngh1 = neigh_split + (l0 << 6);
      int t = -1;
      int k = ngh0[pi0[j]];
      if (k != -1) {
        int l1 = children[k];
        if (l1 != -1) {
          t = (l1 << 3) + dis[j];
        }
      }
      ngh1[j] = t;
    }
  }
}

void calc_neigh_gpu(int* neigh_split, const int* neigh,  const int* children,
    const int node_num, const int* parent, const int* dis) {
  int n = node_num << 6; // node_num: the non_empty node number of parent layer
  calc_neigh_kernel <<< CudaGetBlocks(n), kCudaThreadsNum >>> (
      neigh_split, neigh, children, parent, dis, n);
}

__global__ void calc_full_neigh_kernel(int* neigh, const int depth,
    const int batch_size, const int thread_num) {
  CUDA_KERNEL_LOOP(id, thread_num) {
    const uintk bit = 1;
    const uintk  bound = 1 << depth;
    uintk node_num = bit << 3 * depth;
    uintk num = node_num >> 3;

    uintk tm = id;
    uintk z = tm % 4; tm /= 4;
    uintk y = tm % 4; tm /= 4;
    uintk x = tm % 4; tm /= 4;
    uintk i = (tm % num) * 8;
    uintk n = tm / num;

    uintk x0 = 0, y0 = 0, z0 = 0;
#pragma unroll 4
    for (uintk d = 0; d < depth; d++) {
      x0 |= (i & (bit << 3 * d + 2)) >> (2 * d + 2);
      y0 |= (i & (bit << 3 * d + 1)) >> (2 * d + 1);
      z0 |= (i & (bit << 3 * d + 0)) >> (2 * d + 0);
    }

    uintk x1 = x0 + x - 1;
    uintk y1 = y0 + y - 1;
    uintk z1 = z0 + z - 1;

    int v = -1;
    if ((x1 & bound) == 0 &&
        (y1 & bound) == 0 &&
        (z1 & bound) == 0) {
      uintk key1 = 0;
#pragma unroll 4
      for (int d = 0; d < depth; d++) {
        uintk mask = 1u << d;
        key1 |= ((x1 & mask) << (2 * d + 2)) |
            ((y1 & mask) << (2 * d + 1)) |
            ((z1 & mask) << (2 * d));
      }
      v = key1 + n * node_num;
    }

    neigh[id] = v;
  }
}

void calc_neigh_gpu(int* neigh, const int depth, const int batch_size) {
  int thread_num = batch_size * (1 << 3 * depth + 3);
  calc_full_neigh_kernel <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      neigh, depth, batch_size, thread_num);
  CUDA_POST_KERNEL_CHECK;
}


template<typename Dtype>
__global__ void gen_key_kernel(Dtype* key_child, const Dtype* key,
    const int* child, const int thread_num) {
  typedef typename KeyTrait<Dtype>::uints T;
  CUDA_KERNEL_LOOP(id, thread_num) {
    int i = id >> 3;
    int j = id % 8;

    int label = child[i];
    if (label != -1) {
      const T* k0 = (const T*)(key + i);
      T* k1 = (T*)(key_child + 8 * label + j);
      k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
      k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
      k1[2] = (k0[2] << 1) | (j & 1);
      k1[3] =  k0[3];
    }
  }
}

// use the information from parent layer to calculate the key of current layer
template<typename Dtype>
void generate_key_gpu(Dtype* key_child, const Dtype* key, const int* child,
    const int node_num) {
  int n = node_num << 3; // node_num: the node number of parent layer
  gen_key_kernel <<< CudaGetBlocks(n), kCudaThreadsNum >>> (
      key_child, key, child, n);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void gen_full_key_kernel(Dtype* key, const int depth,
    const int batch_size, const int thread_num) {
  typedef typename KeyTrait<Dtype>::uints T;
  const Dtype bit = 1;
  CUDA_KERNEL_LOOP(i, thread_num) {
    Dtype node_num = bit << 3 * depth;
    Dtype k = i % node_num;
    Dtype xyz = 0;
    T* ptr = (T*)(&xyz);
#pragma unroll 8
    for (int d = 0; d < depth; d++) {
      ptr[0] |= (k & (bit << (3 * d + 2))) >> (2 * d + 2);
      ptr[1] |= (k & (bit << (3 * d + 1))) >> (2 * d + 1);
      ptr[2] |= (k & (bit << (3 * d + 0))) >> (2 * d + 0);
    }
    ptr[3] = i / node_num;
    key[i] = xyz;
  }
}

template<typename Dtype>
void generate_key_gpu(Dtype* key, const int depth, const int batch_size) {
  int thread_num = batch_size * (1 << 3 * depth);
  gen_full_key_kernel <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      key, depth, batch_size, thread_num);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void generate_label_gpu(int* label_data, int& top_h, const Dtype* btm_data,
    const int btm_h, const int mask) {
  top_h = 0;
  thrust::transform_exclusive_scan(thrust::device, btm_data, btm_data + btm_h,
      label_data, mask == thrust::placeholders::_1, 0, thrust::plus<int>());
  cudaMemcpy(&top_h, label_data + btm_h - 1, sizeof(int), cudaMemcpyDeviceToHost);
  Dtype flag = -1;
  cudaMemcpy(&flag, btm_data + btm_h - 1, sizeof(Dtype), cudaMemcpyDeviceToHost);
  if (mask == flag) top_h++;
  thrust::replace_if(thrust::device, label_data, label_data + btm_h, btm_data,
      mask != thrust::placeholders::_1, -1);
}


__global__ void bilinear_neigh_kernel(int* bidx, const int* neigh, const int* child,
    const int node_num, const int* table) {
  CUDA_KERNEL_LOOP(i, node_num) {
    int cld = child[i];
    if (cld < 0) continue;    // skip empty node
    const int* nghi = neigh + (i >> 3 << 6);
#pragma unroll 8
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


void bilinear_neigh_gpu(int* bidx, const int* neigh, const int* child,
    const int node_num, const int* table) {
  bilinear_neigh_kernel <<< CudaGetBlocks(node_num), kCudaThreadsNum >>> (
      bidx, neigh, child, node_num, table);
  CUDA_POST_KERNEL_CHECK;
}


__global__ void bilinear_xyz_kernel(uintk* xyz0, float* fracs,
    const uintk* xyz1, const float scale, const int num) {
  typedef typename KeyTrait<uintk>::uints uints;
  const int mask[8][3] = {                       // bilinear mask:
    {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0},  // 27, 9, 9, 9
    {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},  //  3, 3, 3, 1
  };

  CUDA_KERNEL_LOOP(i, num) {
    float pt[3] = { 0.0f };
    float* frac = fracs + 3 * i;
    int bnd[2][3] = { 0 };
    const uints* ptr1 = (const uints*)(xyz1 + i);
#pragma unroll 3
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

#pragma unroll 8
    for (int j = 0; j < 8; ++j) {
      uints* ptr0 = (uints*)(xyz0 + i * 8 + j);
      for (int c = 0; c < 3; ++c) {
        ptr0[c] = static_cast<uints>(bnd[mask[j][c]][c]);
      }
      ptr0[3] = ptr1[3];
    }
  }
}


void bilinear_xyz_gpu(uintk* xyz0, float* fracs, const int d0, const uintk* xyz1,
    const int d1, const int num) {
  const float scale = static_cast<float>(1 << (d1 - d0));
  bilinear_xyz_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      xyz0, fracs, xyz1, scale, num);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void sequence_gpu(Dtype* ptr, const int num) {
  thrust::sequence(thrust::device, ptr, ptr + num);
}


template <typename Dtype>
__global__ void validate_search_kernel(int* idx, const Dtype* key, const int n_key,
    const Dtype* query, const int n_query) {
  CUDA_KERNEL_LOOP(i, n_query) {
    int j = idx[i];
    if (j >= n_key || key[j] != query[i]) idx[i] = -1;
  }
}

template <typename Dtype>
void search_key_gpu(int* idx, const Dtype* key, const int n_key,
    const Dtype* query, const int n_query) {
  thrust::lower_bound(thrust::device, key, key + n_key, query, query + n_query, idx);
  validate_search_kernel <<< CudaGetBlocks(n_query), kCudaThreadsNum >>> (
      idx, key, n_key, query, n_query);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void xyz2key_kernel(Dtype* key, const Dtype* xyz,
    const int num, const int depth) {
  typedef typename KeyTrait<Dtype>::uints T;

  CUDA_KERNEL_LOOP(i, num) {
    Dtype xyz_in = xyz[i];
    Dtype key_out = 0;
    T* ptr = (T*)(&xyz_in);
    T* ptr_out = (T*)(&key_out);
#pragma unroll 8
    for (int d = 0; d < depth; ++d) {
      T mask = 1 << d;
      key_out |= Dtype(ptr[0] & mask) << (2 * d + 2) |
                 Dtype(ptr[1] & mask) << (2 * d + 1) |
                 Dtype(ptr[2] & mask) << (2 * d + 0);
    }
    ptr_out[3] = ptr[3];
    key[i] = key_out;
  }
}

template <typename Dtype>
void xyz2key_gpu(Dtype* key, const Dtype* xyz, const int num, const int depth) {
  xyz2key_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      key, xyz, num, depth);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void key2xyz_kernel(Dtype* xyz, const Dtype* key,
    const int num, const int depth) {
  typedef typename KeyTrait<Dtype>::uints T;
  const Dtype bit = 1;

  CUDA_KERNEL_LOOP(i, num) {
    Dtype key_in = key[i], xyz_out = 0;
    T* pt = (T*)(&xyz_out);
    T* ptr = (T*)(&key_in);
    pt[3] = ptr[3];
#pragma unroll 8
    for (int d = 0; d < depth; d++) {
      pt[0] |= (key_in & (bit << (3 * d + 2))) >> (2 * d + 2);
      pt[1] |= (key_in & (bit << (3 * d + 1))) >> (2 * d + 1);
      pt[2] |= (key_in & (bit << (3 * d))) >> (2 * d);
    }

    xyz[i] = xyz_out;
  }
}

template <typename Dtype>
void key2xyz_gpu(Dtype* xyz, const Dtype* key, const int num, const int depth) {
  key2xyz_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      xyz, key, num, depth);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void key2idx_kernel(int* idx, const Dtype* key, const int num) {
  typedef typename KeyTrait<Dtype>::uints T;

  CUDA_KERNEL_LOOP(i, num) {
    const T* ptr = (const T*)(key + i);
    idx[i] = static_cast<int>(ptr[3]);
  }
}

template<typename Dtype>
void key2idx_gpu(int* idx, const Dtype* key, const int num) {
  key2idx_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (idx, key, num);
  CUDA_POST_KERNEL_CHECK;
}


template<typename Dtype>
__global__ void xyz2coord_kernel(float* pt, const Dtype* xyz, const int num,
    const int nthreads) {
  typedef typename KeyTrait<Dtype>::uints T;

  CUDA_KERNEL_LOOP(i, nthreads) {
    int h = i % num, c = i / num;
    const T* ptr = (const T*)(xyz + h);
    pt[i] = static_cast<float>(ptr[c]);
  }
}

template<typename Dtype>
void xyz2coord_gpu(float* pt, const Dtype* xyz, const int num, const int channel) {
  int nthreads = num * channel;
  xyz2coord_kernel <<< CudaGetBlocks(nthreads), kCudaThreadsNum >>> (
      pt, xyz, num, nthreads);
  CUDA_POST_KERNEL_CHECK;
}


template<typename Dtype>
__global__ void coord2xyz_kernel(Dtype* xyz, const float* pt, const int num,
    const int nthreads) {
  typedef typename KeyTrait<Dtype>::uints T;

  CUDA_KERNEL_LOOP(i, nthreads) {
    int h = i % num, c = i / num;
    T* ptr = (T*)(xyz + h);
    ptr[c] = static_cast<T>(pt[i]);
  }
}

template<typename Dtype>
void coord2xyz_gpu(Dtype* xyz, const float* pt, const int num, const int channel) {
  int nthreads = num * channel;
  coord2xyz_kernel <<< CudaGetBlocks(nthreads), kCudaThreadsNum >>> (
      xyz, pt, num, nthreads);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void align_forward_kernel(Dtype* top_data, const int top_h,
    const Dtype* btm_data, const int btm_h, const int* index_data,
    const int btm_num) {
  CUDA_KERNEL_LOOP(i, btm_num) {
    int h = i % btm_h;
    int c = i / btm_h;
    int j = index_data[h];
    if (j != -1) {
      top_data[c * top_h + j] = btm_data[i];
    }
  }
}

template <typename Dtype>
void align_forward_gpu(Dtype* top_data, const int top_h, const int channel,
    const Dtype* btm_data, const int btm_h, const int* idx) {
  int btm_num = btm_h * channel;
  memset_gpu(top_h * channel, Dtype(0), top_data);
  align_forward_kernel <<< CudaGetBlocks(btm_num), kCudaThreadsNum >>> (
      top_data, top_h, btm_data, btm_h, idx, btm_num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void align_backward_kernel(const Dtype* top_data, const int top_h,
    Dtype* btm_data, const int btm_h, const int* index_data, const int btm_num) {
  CUDA_KERNEL_LOOP(i, btm_num) {
    int h = i % btm_h;
    int c = i / btm_h;
    int j = index_data[h];
    btm_data[i] = j == -1 ? 0 : top_data[c * top_h + j];
  }
}

template <typename Dtype>
void align_backward_gpu(const Dtype* top_data, const int top_h, const int channel,
    Dtype* btm_data, const int btm_h, const int* idx) {
  int btm_num = btm_h * channel;
  align_backward_kernel <<< CudaGetBlocks(btm_num), kCudaThreadsNum >>> (
      top_data, top_h, btm_data, btm_h, idx, btm_num);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void octree_gather_kernel(Dtype* top_data, const int top_h,
    const Dtype* btm_data, const int btm_h, const int* index_data, const int num) {
  CUDA_KERNEL_LOOP(i, num) {
    int h = i % top_h;
    int c = i / top_h;
    int j = index_data[h];
    if (j != -1) {
      top_data[i] = btm_data[c * btm_h + j];
    }
  }
}

template <typename Dtype>
void octree_gather_gpu(Dtype* top_data, const int top_h, const int channel,
    const Dtype* btm_data, const int btm_h, const int* idx) {
  pad_forward_gpu<Dtype>(top_data, top_h, channel, btm_data, btm_h, idx, Dtype(0));

  //int num = top_h * channel;
  //memset_gpu(num, Dtype(0), top_data);
  //octree_gather_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
  //    top_data, top_h, btm_data, btm_h, idx, num);
  //CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void octree_gatherbk_kernel(const Dtype* top_data, const int top_h,
    Dtype* btm_data, const int btm_h, const int* index_data, const int num) {
  CUDA_KERNEL_LOOP(i, num) {
    int h = i % top_h;
    int c = i / top_h;
    int j = index_data[h];
    if (j != -1) {
      caffe_gpu_atomic_add(top_data[i], btm_data + c * btm_h + j);
    }
  }
}

template <typename Dtype>
void octree_gatherbk_gpu(const Dtype* top_data, const int top_h, const int channel,
    Dtype* btm_data, const int btm_h, const int* idx) {
  int num = top_h * channel;
  memset_gpu(channel * btm_h, Dtype(0), btm_data);
  octree_gatherbk_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      top_data, top_h, btm_data, btm_h, idx, num);
  CUDA_POST_KERNEL_CHECK;
}



__global__ void octree_mask_kernel(float* des, const float* src,
    const int* label_data, const int height, const int mask, const int n) {
  CUDA_KERNEL_LOOP(i, n) {
    int h = i % height;
    des[i] = label_data[h] == mask ? float(0) : src[i];
  }
}

void octree_mask_gpu(float* out_data, const float* in_data, const int* label,
    int height, int mask, int num) {
  octree_mask_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      out_data, in_data, label, height, mask, num);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void memset_gpu<int>(const size_t N, const int alpha, int* Y);
template void memset_gpu<float>(const size_t N, const float alpha, float* Y);
template void memset_gpu<double>(const size_t N, const double alpha, double* Y);
template void memset_gpu<char>(const size_t N, const char alpha, char* Y);
template void memset_gpu<int8_t>(const size_t N, const int8_t alpha, int8_t* Y);
template void memset_gpu<uint8_t>(const size_t N, const uint8_t alpha, uint8_t* Y);
template void memcpy_gpu<char>(const size_t N, const char* X, char* Y);
template void memcpy_gpu<int>(const size_t N, const int* X, int* Y);
template void memcpy_gpu<int64_t>(const size_t N, const int64_t* X, int64_t* Y);
template void memcpy_gpu<int16_t>(const size_t N, const int16_t* X, int16_t* Y);
template void memcpy_gpu<uint32>(const size_t N, const uint32* X, uint32* Y);
template void memcpy_gpu<uint64>(const size_t N, const uint64* X, uint64* Y);
template void memcpy_gpu<float>(const size_t N, const float* X, float* Y);
template void memcpy_gpu<double>(const size_t N, const double* X, double* Y);
template void sequence_gpu<int>(int* ptr, const int num);
template void sequence_gpu<uintk>(uintk* ptr, const int num);
template void pad_forward_gpu<float>(float* Y, const int Hy, const int Cy,
    const float* X, const int Hx, const int* label, const float dval);
template void pad_forward_gpu<double>(double* Y, const int Hy, const int Cy,
    const double* X, const int Hx, const int* label, const double dval);
template void pad_backward_gpu<float>(float* X, const int Hx, const int Cx,
    const float* Y, const int Hy, const int* label);
template void pad_backward_gpu<double>(double* X, const int Hx, const int Cx,
    const double* Y, const int Hy, const int* label);
template void pad_backward_gpu<int>(int* X, const int Hx, const int Cx,
    const int* Y, const int Hy, const int* label);
template void pad_backward_gpu<uintk>(uintk* X, const int Hx, const int Cx,
    const uintk* Y, const int Hy, const int* label);
template void octree2col_gpu<float>(float* data_col, const float* data_octree,
    const int channel, const int height,  const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void octree2col_gpu<double>(double* data_col, const double* data_octree,
    const int channel, const int height, const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void col2octree_gpu<float>(const float* data_col, float* data_octree,
    const int channel, const int height, const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void col2octree_gpu<double>(const double* data_col, double* data_octree,
    const int channel, const int height, const int kernel_sdim, const int stride,
    const int* neigh, const int* ni, const int height_col, const int n);
template void octree2colP_gpu<float>(float* data_col, const float* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
template void col2octreeP_gpu<float>(const float* data_col, float* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
template void octree2colP_gpu<double>(double* data_col, const double* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
template void col2octreeP_gpu<double>(const double* data_col, double* data_octree, 
    const int channel, const int height, const int octree_h, const int kernel_sdim, 
    const int stride, const int* neigh, const int* ni, const int* child, 
    const int* ichild, const int height_col, const int n);
template void generate_label_gpu<float>(int* label_data, int& top_h,
    const float* bottom_data, const int bottom_h, const int mask);
template void generate_label_gpu<double>(int* label_data, int& top_h,
    const double* bottom_data, const int bottom_h, const int mask);
template void generate_label_gpu<int>(int* label_data, int& top_h,
    const int* bottom_data, const int bottom_h, const int mask);
template void octree_max_pool_gpu<float>(float* top_data, int top_h,
    int* mask, const float* btm_data, int bottom_h, int channel);
template void octree_max_pool_gpu<double>(double* top_data, int top_h,
    int* mask, const double* btm_data, int bottom_h, int channel);
template void octree_max_unpool_gpu<float>(const float* top_data, int top_h,
    const int* mask, float* btm_data, int bottom_h, int channel);
template void octree_max_unpool_gpu<double>(const double* top_data, int top_h,
    const int* mask, double* btm_data, int bottom_h, int channel);
template void octree_mask_pool_gpu<float>(float* top_data, int top_h,
    const int* mask, const float* btm_data, int bottom_h, int channel);
template void octree_mask_pool_gpu<double>(double* top_data, int top_h,
    const int* mask, const double* btm_data, int bottom_h, int channel);
template void align_forward_gpu(float* top_data, const int top_h, const int c,
    const float* btm_data, const int btm_h, const int* idx);
template void align_forward_gpu(double* top_data, const int top_h, const int c,
    const double* btm_data, const int btm_h, const int* idx);
template void align_backward_gpu(const float* top_data, const int top_h,
    const int c, float* btm_data, const int btm_h, const int* idx);
template void align_backward_gpu(const double* top_data, const int top_h,
    const int c, double* btm_data, const int btm_h, const int* idx);
template void octree_gather_gpu(float* top_data, const int top_h, const int c,
    const float* btm_data, const int btm_h, const int* idx);
template void octree_gather_gpu(double* top_data, const int top_h, const int c,
    const double* btm_data, const int btm_h, const int* idx);
template void octree_gatherbk_gpu(const float* top_data, const int top_h,
    const int c, float* btm_data, const int btm_h, const int* idx);
template void octree_gatherbk_gpu(const double* top_data, const int top_h,
    const int c, double* btm_data, const int btm_h, const int* idx);
template void generate_key_gpu<uintk>(uintk* key, const int depth, const int batch_size);
template void generate_key_gpu<uintk>(uintk* key_child, const uintk* key,
  const int* child, const int node_num);
template void search_key_gpu<uintk>(int* idx, const uintk* key, const int n_key,
  const uintk* query, const int n_query);
template void xyz2key_gpu<uintk>(uintk* key, const uintk* xyz, const int num,
  const int depth);
template void key2xyz_gpu<uintk>(uintk* xyz, const uintk* key, const int num,
  const int depth);
template void key2idx_gpu<uintk>(int* idx, const uintk* key, const int num);
template void xyz2coord_gpu<uintk>(float* pt, const uintk* xyz, const int num,
  const int channel);
template void coord2xyz_gpu<uintk>(uintk* xyz, const float* pt, const int num,
  const int channel);
