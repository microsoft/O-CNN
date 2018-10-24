#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/octree2col_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Octree2ColLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // kernel size
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  size_t kss = conv_param.kernel_size_size();
  CHECK(kss == 1 || kss == 3) << "Specify 1 or 3 kernel_size paramters";
  kernel_size_.resize(3);
  if (kss == 1) {
    kernel_size_[0] = conv_param.kernel_size(0);
    kernel_size_[1] = conv_param.kernel_size(0);
    kernel_size_[2] = conv_param.kernel_size(0);
  } else {
    kernel_size_[0] = conv_param.kernel_size(0);
    kernel_size_[1] = conv_param.kernel_size(1);
    kernel_size_[2] = conv_param.kernel_size(2);
  }
  CHECK(kernel_size_[0] < 4 && kernel_size_[1] < 4 && kernel_size_[2] < 4)
      << "kernel_size should be less than 4";
  kernel_sdim_ = kernel_size_[0] * kernel_size_[1] * kernel_size_[2];

  // stride
  CHECK_LE(conv_param.stride_size(), 1);
  stride_ = conv_param.stride_size() == 0 ? 1 : conv_param.stride(0);
  CHECK_LE(stride_, 2) << "stride should be less than 2";

  // special case: im2col is the identity for 1x1 convolution with stride 1
  is_1x1_ = kernel_size_[0] == 1 && kernel_size_[1] == 1 &&
      kernel_size_[2] == 1 && stride_ == 1;

  // current octree depth
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
          << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();

  // channels & kernel_dim_
  channels_ = bottom[0]->shape(1);
  kernel_dim_ = channels_ * kernel_sdim_;
}

template <typename Dtype>
void Octree2ColLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // a workaround for the first time reshape
  if (top[0]->count() == 0) {
    top[0]->Reshape(vector<int> {1, kernel_dim_, 8, 1});
    return;
  }

  bool octree_in = bottom.size() == 2;
  Blob<Dtype>& the_octree = octree_in ? *bottom[1] : Octree::get_octree(Dtype(0));
  octree::set_octree_parser(octree_batch_, the_octree);

  // check bottom shape
  int bottom_h = bottom[0]->shape(2);
  CHECK_EQ(bottom_h, octree_batch_.info().node_num(curr_depth_))
      << "Error in " << this->layer_param_.name();

  // reshape top blob
  int top_h = (stride_ == 2) ? (bottom_h >> 3) : bottom_h;
  top[0]->Reshape(vector<int> { 1, kernel_dim_, top_h, 1 });
}

template<typename Dtype>
void Octree2ColLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int top_h = top[0]->shape(2);

  if (is_1x1_) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
    return;
  }

  octree::octree2col_cpu<Dtype>(top_data,	bottom_data, channels_, top_h,
      kernel_sdim_, stride_, octree_batch_.neighbor_cpu(curr_depth_),
      Octree::get_ni(kernel_size_)->cpu_data(), top_h, 0);
}

template<typename Dtype>
void caffe::Octree2ColLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int top_h = top[0]->shape(2);
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (is_1x1_) {
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
    return;
  }

  octree::col2octree_cpu<Dtype>(top_diff, bottom_diff, channels_, top_h,
      kernel_sdim_, stride_, octree_batch_.neighbor_cpu(curr_depth_),
      Octree::get_ni(kernel_size_)->cpu_data(), top_h, 0);
}


#ifdef CPU_ONLY
STUB_GPU(Octree2ColLayer);
#endif

REGISTER_LAYER_CLASS(Octree2Col);
INSTANTIATE_CLASS(Octree2ColLayer);

}  // namespace caffe
