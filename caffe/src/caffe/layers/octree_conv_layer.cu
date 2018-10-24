#include "caffe/layers/octree_conv_layer.hpp"
#include "caffe/util/gpu_util.cuh"


namespace caffe {
template <typename Dtype>
void OctreeConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // gemm
  Dtype* top_data = this->stride_ == 1 ? top[0]->mutable_gpu_data() :
      this->data_buffer_->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weights = this->blobs_[0]->gpu_data();
  this->forward_gpu_gemm(top_data, bottom_data, weights);

  // bias
  if (this->bias_term_) {
    const Dtype* bias = this->blobs_[1]->gpu_data();
    this->forward_gpu_bias(top_data, bias);
  }

  // pad
  if (this->stride_ == 2) {
    octree::pad_forward_gpu<Dtype>(top[0]->mutable_gpu_data(),
        top[0]->shape(2), top[0]->shape(1),
        this->data_buffer_->gpu_data(), this->data_buffer_shape_[2],
        this->octree_batch_.children_gpu(this->curr_depth_ - 1));
  }
}

template <typename Dtype>
void OctreeConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->stride_ == 2) {
    octree::pad_backward_gpu(this->data_buffer_->mutable_gpu_data(),
        this->data_buffer_shape_[2], this->data_buffer_shape_[1],
        top[0]->gpu_diff(), top[0]->shape(2),
        this->octree_batch_.children_gpu(this->curr_depth_ - 1));
  }

  // Bias gradient, if necessary.
  const Dtype* top_diff = this->stride_ == 1 ?
      top[0]->gpu_diff() : this->data_buffer_->gpu_data();
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    this->backward_gpu_bias(bias_diff, top_diff);
  }

  if (propagate_down[0] || this->param_propagate_down_[0]) {
    // gradient w.r.t. weight, if necessary
    if (this->param_propagate_down_[0]) {
      Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      this->weight_gpu_gemm(weight_diff, bottom_data, top_diff);
    }

    // gradient w.r.t. bottom data, if necessary
    if (propagate_down[0]) {
      const Dtype* weights = this->blobs_[0]->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      this->backward_gpu_gemm(bottom_diff, top_diff, weights);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeConvLayer);
}