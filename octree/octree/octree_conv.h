#ifndef _OCTREE_OCTREE_BASE_CONV_
#define _OCTREE_OCTREE_BASE_CONV_

#include <vector>
#include "gemm_engine.h"
#include "octree_parser.h"

namespace octree {

using std::vector;

template <typename Dtype>
class OctreeBaseConv {
 public:
  explicit OctreeBaseConv(int max_size = 256 * 1024 * 1024)
      : MAX_SIZE(max_size), engine_cpu_(nullptr), engine_gpu_(nullptr),
        nempty_(false), child_(nullptr), ichild_(nullptr) {}
  void setup(const vector<int>& kernel_size, const int stride,
      const int curr_depth, const int channel_in, const int channel_out,
      const bool nempty = false);
  // !!! Please set engine_cpu/gpu_, octree_ and ni_gpu_ptr_
  // after calling setup() and before reshpae(),
  void reshape();

 protected:
  // return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool is_deconvolution_layer() = 0;

  // Helper functions that abstract away the column buffer and gemm arguments.
  void forward_cpu_gemm(Dtype* top_data, const Dtype* bottom_data,
      const Dtype* weights);
  void backward_cpu_gemm(Dtype* bottom_diff, const Dtype* top_diff,
      const Dtype* weights);
  void weight_cpu_gemm(Dtype* weights_diff, const Dtype* bottom_data,
      const Dtype* top_diff);

  void forward_gpu_gemm(Dtype* top_data, const Dtype* bottom_data,
      const Dtype* weights);
  void backward_gpu_gemm(Dtype* bottom_diff, const Dtype* top_diff,
      const Dtype* weights);
  void weight_gpu_gemm(Dtype* weights_diff, const Dtype* bottom_data,
      const Dtype* top_diff);

  void octree2col_cpu_wrapper(Dtype* workspace, const Dtype* bottom_data, int n);
  void col2octree_cpu_wrapper(const Dtype* workspace, Dtype* bottom_data, int n);
  void octree2col_gpu_wrapper(Dtype* workspace, const Dtype* bottom_data, int n);
  void col2octree_gpu_wrapper(const Dtype* workspace, Dtype* bottom_data, int n);

 protected:
  int stride_;
  vector<int> kernel_size_;
  int kernel_dim_;
  int kernel_sdim_;  // spatial dim of the kernel
  bool is_1x1_;

  // input channel & output channel
  int channels_;
  int num_output_;

  // helper channels for conv and deconv
  int conv_out_channels_;
  int conv_in_channels_;

  int curr_depth_;
  OctreeParser octree_;

  int workspace_n_;
  int workspace_ha_;    // actual worksapce h, the height of `col` data
  int workspace_h_;     // ideal  workspace h
  int workspace_depth_; // the depth value used for octree2col

  vector<int> top_shape_;
  vector<int> weights_shape_;
  vector<int> workspace_shape_;
  vector<int> result_buffer_shape_;

  Dtype* workspace_;
  Dtype* result_buffer_;  // hold the temporary result of octree2col

  const int* ni_cpu_ptr_; // hold cpu data from NeighHelper::get_ni(kernel_size_)
  const int* ni_gpu_ptr_; // hold gpu data from NeighHelper::get_ni(kernel_size_)

  uint64 MAX_SIZE;

  GEMMEngine<Dtype>* engine_cpu_;
  GEMMEngine<Dtype>* engine_gpu_;

  bool nempty_;          // perform convolution on non-empty voxels

  // used for octree2col and col2octree on non-empty voxels
  int octree_h_;         // the height of octree data
  int child_h_;
  int ichild_h_;
  const int* child_;
  const int* ichild_;
};

}  // namespace octree

#endif  // _OCTREE_OCTREE_BASE_CONV_
