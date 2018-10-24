#include "caffe/layers/octree_depadding_layer.hpp"
#include "caffe/util/gpu_util.cuh"
namespace caffe {
template <typename Dtype>
void OctreeDepaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int channel = top[0]->shape(1);
  int top_h = top[0]->shape(2);
  int bottom_h = bottom[0]->shape(2);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (top_h != bottom_h) {
    octree::pad_backward_gpu<Dtype>(top_data, top_h, channel,
        bottom_data, bottom_h, octree_batch_.children_gpu(curr_depth_));
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void OctreeDepaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  // padding
  int channel = top[0]->shape(1);
  int top_h = top[0]->shape(2);
  int bottom_h = bottom[0]->shape(2);
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (top_h != bottom_h) {
    octree::pad_forward_gpu<Dtype>(bottom_diff, bottom_h, channel,
        top_diff, top_h, octree_batch_.children_gpu(curr_depth_));
  } else {
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeDepaddingLayer);
}