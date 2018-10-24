#include "caffe/layers/octree2col_layer.hpp"
#include "caffe/util/octree.hpp"

namespace caffe {

template<typename Dtype>
void Octree2ColLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int top_h = top[0]->shape(2);
  if (is_1x1_) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
    return;
  }

  octree::octree2col_gpu<Dtype>(top_data, bottom_data, channels_, top_h,
      kernel_sdim_, stride_, octree_batch_.neighbor_gpu(curr_depth_),
      Octree::get_ni(kernel_size_)->gpu_data(), top_h, 0);
}

template<typename Dtype>
void caffe::Octree2ColLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int top_h = top[0]->shape(2);
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (is_1x1_) {
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
    return;
  }

  octree::col2octree_gpu<Dtype>(top_diff, bottom_diff, channels_, top_h,
      kernel_sdim_, stride_, octree_batch_.neighbor_gpu(curr_depth_),
      Octree::get_ni(kernel_size_)->gpu_data(), top_h, 0);
}

INSTANTIATE_LAYER_GPU_FUNCS(Octree2ColLayer);

}