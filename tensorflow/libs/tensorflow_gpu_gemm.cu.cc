#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "tensorflow_gpu_gemm.h"

#include <cuda.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/stream_executor.h>

namespace tensorflow {

namespace {
perftools::gputools::DeviceMemory<float> AsDeviceMemory(
    const float* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(
      const_cast<float*>(cuda_memory));
  perftools::gputools::DeviceMemory<float> typed(wrapped);
  return typed;
}
} // namespace

void GEMMEngineTF::gemm(const bool transa, const bool transb,
    const int m, const int n, const int k, const float alpha,
    const float* a, const float* b, const float beta, float* c) {
  perftools::gputools::blas::Transpose trans[] = {
    perftools::gputools::blas::Transpose::kNoTranspose,
    perftools::gputools::blas::Transpose::kTranspose
  };

  auto* stream = context_->op_device_context()->stream();
  OP_REQUIRES(context_, stream, errors::Internal("No GPU stream available."));

  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);

  bool blas_launch_status =
      stream
      ->ThenBlasGemm(trans[transb], trans[transa], n, m, k, alpha, b_ptr,
          transb ? k : n, a_ptr, transa ? m : k, beta, &c_ptr, n)
      .ok();

  OP_REQUIRES(context_, blas_launch_status, errors::Aborted("CuBlasGemm failed!"));
}

} // namespace tensorflow