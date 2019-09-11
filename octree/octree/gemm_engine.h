#ifndef _OCTREE_GEMM_ENGINE_
#define _OCTREE_GEMM_ENGINE_


namespace octree {

template <typename Dtype>
class GEMMEngine {
 public:
  // C = beta * C + alpha * A * B
  virtual void gemm(const bool TransA, const bool TransB,
      const int M, const int N, const int K, const Dtype alpha,
      const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C) = 0;
};

} // namespace octree

#endif // _OCTREE_GEMM_ENGINE_
