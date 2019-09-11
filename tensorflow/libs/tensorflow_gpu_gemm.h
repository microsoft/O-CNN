#ifndef _TENSORFLOW_GPU_GEMM_H_
#define _TENSORFLOW_GPU_GEMM_H_

#include "gemm_engine.h"

namespace tensorflow {

class OpKernelContext;

class GEMMEngineTF : public octree::GEMMEngine<float> {
 public:
  void set_context(OpKernelContext* ctx) { context_ = ctx; }
  
  virtual void gemm(const bool TransA, const bool TransB,
      const int M, const int N, const int K, const float alpha,
      const float* A, const float* B, const float beta, float* C) override;

 public:
  OpKernelContext* context_;
};

} // namespace tensorflow

#endif // _TENSORFLOW_GPU_GEMM_H_
