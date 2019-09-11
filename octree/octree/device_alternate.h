#ifndef OCTREE_DEVICE_ALTERNATE_H_
#define OCTREE_DEVICE_ALTERNATE_H_

#include "logs.h"

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>


// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK(error == cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())


// CUDA: use 512 threads per block
const int kCudaThreadsNum = 512;

// CUDA: number of blocks for threads.
inline int CudaGetBlocks(const int N) {
  return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}


#else

#define NO_GPU CHECK(false) << "Cannot use GPU in CPU mode."

#endif  // USE_CUDA

#endif  // OCTREE_DEVICE_ALTERNATE_H_
