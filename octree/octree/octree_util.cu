#include "octree_util.h"
#include "device_alternate.h"

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
void memset_gpu(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));
    return;
  }
  memset_kernel<Dtype> <<< CudaGetBlocks(N), kCudaThreadsNum >>> (
      N, alpha, Y);
}

template <typename Dtype>
void memcpy_gpu(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
  }
}


template <typename Dtype>
__global__ void pad_forward_kernel(Dtype* Y, const int Hy,
    const Dtype* X, const int Hx, const int* label, const int n) {
  CUDA_KERNEL_LOOP(i, n) {
    int h = i % Hy;
    int c = i / Hy;

    int idx = label[h];
    Y[i] = idx == -1 ? Dtype(0) : X[c * Hx + idx];
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
    const Dtype* X, const int Hx, const int* label) {
  int n = Hy * Cy; // Note: Cx == Cy
  pad_forward_kernel<Dtype> <<< CudaGetBlocks(n), kCudaThreadsNum >>> (
      Y, Hy, X, Hx, label, n);
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
  const int kernel = kernel_sdim;
  const int thread_num = channel * kernel * height_col;
  octree2col_kernel<Dtype> <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      data_col, data_octree, height, kernel, stride, neigh, ni, height_col, n, thread_num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void col2octree_gpu(const Dtype* data_col, Dtype* data_octree,
    const int channel, const int height, const int kernel_sdim,
    const int stride, const int* neigh, const int* ni,
    const int height_col, const int n) {
  const int kernel = kernel_sdim; // kernel size: 3*3*3
  const int thread_num = channel * kernel * height_col;
  int octree_h = height << 3 * (stride - 1);
  // set data_octree to zero ONCE when n ==0
  if (n == 0) memset_gpu(channel * octree_h, Dtype(0), data_octree);
  col2octree_kernel<Dtype> <<< CudaGetBlocks(thread_num), kCudaThreadsNum >>> (
      data_col, data_octree, height, kernel, stride, neigh, ni, height_col, n, thread_num);
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
    const unsigned  bound = 1 << depth;
    unsigned node_num = 1 << 3 * depth;
    unsigned num = node_num >> 3;

    unsigned tm = id;
    unsigned z = tm % 4; tm /= 4;
    unsigned y = tm % 4; tm /= 4;
    unsigned x = tm % 4; tm /= 4;
    unsigned i = (tm % num) * 8;
    unsigned n = tm / num;

    unsigned x0 = 0, y0 = 0, z0 = 0;
#pragma unroll 4
    for (unsigned d = 0; d < depth; d++) {
      x0 |= (i & (1 << 3 * d + 2)) >> (2 * d + 2);
      y0 |= (i & (1 << 3 * d + 1)) >> (2 * d + 1);
      z0 |= (i & (1 << 3 * d + 0)) >> (2 * d + 0);
    }

    unsigned x1 = x0 + x - 1;
    unsigned y1 = y0 + y - 1;
    unsigned z1 = z0 + z - 1;

    int v = -1;
    if ((x1 & bound) == 0 &&
        (y1 & bound) == 0 &&
        (z1 & bound) == 0) {
      unsigned key1 = 0;
#pragma unroll 4
      for (int d = 0; d < depth; d++) {
        unsigned mask = 1u << d;
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


__global__ void gen_key_kernel(uint32* key_child, const uint32* key,
    const int* child, const int thread_num) {
  typedef unsigned char ubyte;
  CUDA_KERNEL_LOOP(id, thread_num) {
    int i = id >> 3;
    int j = id % 8;

    int label = child[i];
    if (label != -1) {
      const ubyte* k0 = (const ubyte*)(key + i);
      ubyte* k1 = (ubyte*)(key_child + 8 * label + j);
      k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
      k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
      k1[2] = (k0[2] << 1) | (j & 1);
      k1[3] =  k0[3];
    }
  }
}

// use the information from parent layer to calculate the key of current layer
void generate_key_gpu(uint32* key_child, const uint32* key, const int* child,
    const int node_num) {
  int n = node_num << 3; // node_num: the node number of parent layer
  gen_key_kernel <<< CudaGetBlocks(n), kCudaThreadsNum >>> (
      key_child, key, child, n);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void gen_full_key_kernel(uint32* key, const int depth,
    const int batch_size, const int thread_num) {
  CUDA_KERNEL_LOOP(i, thread_num) {
    unsigned node_num = 1 << 3 * depth;
    unsigned k = i % node_num;
    unsigned xyz = 0;
    unsigned char* ptr = (unsigned char*)(&xyz);
#pragma unroll 8
    for (int d = 0; d < depth; d++) {
      ptr[0] |= (k & (1 << 3 * d + 2)) >> (2 * d + 2);
      ptr[1] |= (k & (1 << 3 * d + 1)) >> (2 * d + 1);
      ptr[2] |= (k & (1 << 3 * d + 0)) >> (2 * d + 0);
    }
    ptr[3] = i / node_num;
    key[i] = xyz;
  }
}

void generate_key_gpu(uint32* key, const int depth, const int batch_size) {
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


template <typename Dtype>
void sequence_gpu(Dtype* ptr, const int num) {
  thrust::sequence(thrust::device, ptr, ptr + num);
}


__global__ void validate_search_kernel(int* idx, const unsigned* key, const int n_key,
    const unsigned* query, const int n_query) {
  CUDA_KERNEL_LOOP(i, n_query) {
    int j = idx[i];
    if (j >= n_key || key[j] != query[i]) idx[i] = -1;
  }
}

void search_key_gpu(int* idx, const uint32* key, const int n_key,
    const uint32* query, const int n_query) {
  thrust::lower_bound(thrust::device, key, key + n_key, query, query + n_query, idx);
  validate_search_kernel <<< CudaGetBlocks(n_query), kCudaThreadsNum >>> (
      idx, key, n_key, query, n_query);
  CUDA_POST_KERNEL_CHECK;
}


// NOTE: !!! currently the depth should be less than 8
__global__ void xyz2key_kernel(uint32* key, const uint32* xyz,
    const int num, const int depth) {
  CUDA_KERNEL_LOOP(i, num) {
    uint32 xyz_in = xyz[i];
    uint32 key_out = 0;
    unsigned char* ptr = (unsigned char*)(&xyz_in);
    unsigned char* ptr_out = (unsigned char*)(&key_out);
#pragma unroll 8
    for (int d = 0; d < depth; ++d) {
      unsigned char mask = 1 << d;
      key_out |= (ptr[0] & mask) << (2 * d + 2) |
          (ptr[1] & mask) << (2 * d + 1) |
          (ptr[2] & mask) << (2 * d + 0);
    }
    ptr_out[3] = ptr[3];
    key[i] = key_out;
  }
}

void xyz2key_gpu(uint32* key, const uint32* xyz, const int num, const int depth) {
  xyz2key_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      key, xyz, num, depth);
  CUDA_POST_KERNEL_CHECK;
}

// NOTE: !!! currently the depth should be less than 8
__global__ void key2xyz_kernel(uint32* xyz, const uint32* key,
    const int num, const int depth) {
  CUDA_KERNEL_LOOP(i, num) {
    uint32 key_in = key[i], xyz_out = 0;
    unsigned char* pt = (unsigned char*)(&xyz_out);
    unsigned char* ptr = (unsigned char*)(&key_in);
    pt[3] = ptr[3];
#pragma unroll 8
    for (int d = 0; d < depth; d++) {
      pt[0] |= (key_in & (1u << (3 * d + 2))) >> (2 * d + 2);
      pt[1] |= (key_in & (1u << (3 * d + 1))) >> (2 * d + 1);
      pt[2] |= (key_in & (1u << (3 * d))) >> (2 * d);
    }

    xyz[i] = xyz_out;
  }
}

void key2xyz_gpu(uint32* xyz, const uint32* key, const int num, const int depth) {
  key2xyz_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      xyz, key, num, depth);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void align_forward_kernel(Dtype* top_data, const int Htop,
    const Dtype* btm_data, const int Hbtm, const int* index_data, const int num) {
  CUDA_KERNEL_LOOP(i, num) {
    int h = i % Hbtm;
    int c = i / Hbtm;
    int j = index_data[h];
    if (j != -1) {
      top_data[c * Htop + j] = btm_data[i];
    }
  }
}

template <typename Dtype>
void align_forward_gpu(Dtype* top_data, const int top_h, const int channel,
    const Dtype* btm_data, const int btm_h, const int* idx, const int num) {
  memset_gpu(num, Dtype(0), top_data);
  align_forward_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
      top_data, top_h, btm_data, btm_h, idx, num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void align_backward_kernel(const Dtype* top_data, const int Htop,
    Dtype* btm_data, const int Hbtm, const int* index_data, const int num) {
  CUDA_KERNEL_LOOP(i, num) {
    int h = i % Hbtm;
    int c = i / Hbtm;
    int j = index_data[h];
    btm_data[i] = j == -1 ? 0 : top_data[c * Htop + j];
  }
}

template <typename Dtype>
void align_backward_gpu(const Dtype* top_data, const int top_h, const int channel,
    Dtype* btm_data, const int btm_h, const int* idx, const int num) {
  align_backward_kernel <<< CudaGetBlocks(num), kCudaThreadsNum >>> (
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
template void memset_gpu<int>(const int N, const int alpha, int* Y);
template void memset_gpu<float>(const int N, const float alpha, float* Y);
template void memset_gpu<double>(const int N, const double alpha, double* Y);
template void memset_gpu<char>(const int N, const char alpha, char* Y);
template void memset_gpu<int8_t>(const int N, const int8_t alpha, int8_t* Y);
template void memset_gpu<uint8_t>(const int N, const uint8_t alpha, uint8_t* Y);
template void memcpy_gpu<int>(const int N, const int* X, int* Y);
template void memcpy_gpu<unsigned>(const int N, const unsigned* X, unsigned* Y);
template void memcpy_gpu<float>(const int N, const float* X, float* Y);
template void memcpy_gpu<double>(const int N, const double* X, double* Y);
template void sequence_gpu<int>(int* ptr, const int num);
template void sequence_gpu<unsigned int>(unsigned int* ptr, const int num);
template void pad_forward_gpu<float>(float* Y, const int Hy, const int Cy,
    const float* X, const int Hx, const int* label);
template void pad_forward_gpu<double>(double* Y, const int Hy, const int Cy,
    const double* X, const int Hx, const int* label);
template void pad_backward_gpu<float>(float* X, const int Hx, const int Cx,
    const float* Y, const int Hy, const int* label);
template void pad_backward_gpu<double>(double* X, const int Hx, const int Cx,
    const double* Y, const int Hy, const int* label);
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
    const float* btm_data, const int btm_h, const int* idx, const int n);
template void align_forward_gpu(double* top_data, const int top_h, const int c,
    const double* btm_data, const int btm_h, const int* idx, const int n);
template void align_backward_gpu(const float* top_data, const int top_h,
    const int c, float* btm_data, const int btm_h, const int* idx, const int n);
template void align_backward_gpu(const double* top_data, const int top_h,
    const int c, double* btm_data, const int btm_h, const int* idx, const int n);