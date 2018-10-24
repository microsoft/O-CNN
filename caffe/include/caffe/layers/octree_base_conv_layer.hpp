#ifndef CAFFE_OCTREE_BASE_CONV_LAYER_HPP_
#define CAFFE_OCTREE_BASE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/octree.hpp"

namespace caffe {

// TODO: use cudnn to speedup
template <typename Dtype>
class OctreeBaseConvLayer : public Layer<Dtype> {
 public:
  explicit OctreeBaseConvLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual ~OctreeBaseConvLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  // return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual inline bool is_deconvolution_layer() = 0;

  // Helper functions that abstract away the column buffer and gemm arguments.
  void forward_cpu_gemm(Dtype* top_data, const Dtype* bottom_data,
      const Dtype* weights);
  void backward_cpu_gemm(Dtype* bottom_diff, const Dtype* top_diff,
      const Dtype* weights);
  void forward_cpu_bias(Dtype* top_data, const Dtype* bias);
  void backward_cpu_bias(Dtype* bias_diff, const Dtype* top_diff);
  void weight_cpu_gemm(Dtype* weights_diff, const Dtype* bottom_data,
      const Dtype* top_diff);

#ifndef CPU_ONLY
  void forward_gpu_gemm(Dtype* top_data, const Dtype* bottom_data,
      const Dtype* weights);
  void backward_gpu_gemm(Dtype* bottom_diff, const Dtype* top_diff,
      const Dtype* weights);
  void forward_gpu_bias(Dtype* top_data, const Dtype* bias);
  void backward_gpu_bias(Dtype* bias_diff, const Dtype* top_diff);
  void weight_gpu_gemm(Dtype* weights_diff, const Dtype* bottom_data,
      const Dtype* top_diff);
#endif

 protected:
  int stride_;
  vector<int> kernel_size_;
  int kernel_dim_;
  int kernel_sdim_; // spatial dim of the kernel
  bool bias_term_;
  bool is_1x1_;

  // input channel & output channel
  int channels_;
  int num_output_;

  // helper channels for conv and deconv
  int conv_out_channels_;
  int conv_in_channels_;

  int curr_depth_;
  OctreeParser octree_batch_;

  int workspace_n_;
  int workspace_ha_;	// actual worksapce h
  int workspace_h_;	// ideal workspace h
  int workspace_depth_;
  int bias_multiplier_h_;
  vector<int> data_buffer_shape_;
  shared_ptr<Blob<Dtype> > workspace_;
  shared_ptr<Blob<Dtype> > data_buffer_;
  shared_ptr<Blob<Dtype> > result_buffer_;
  Blob<Dtype> bias_multiplier_;

#if defined(USE_CUDNN) && !defined(CPU_ONLY)
  // cudnn
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnHandle_t                   handle_;
  cudnnTensorDescriptor_t         bottom_desc_;
  cudnnTensorDescriptor_t         top_desc_;
  cudnnFilterDescriptor_t         filter_desc_;
  cudnnConvolutionDescriptor_t    conv_desc_;
  size_t  filter_workspace_size_;
  Blob<Dtype> filter_workspace_;
#endif // USE_CUDNN
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_BASE_CONV_LAYER_HPP_
