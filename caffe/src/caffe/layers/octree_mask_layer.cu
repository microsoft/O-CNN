#include "caffe/layers/octree_mask_layer.hpp"
#include "caffe/util/gpu_util.cuh"

#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/replace.h>

namespace caffe {

template <typename Dtype>
void OctreeMaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // generate the map
  int top_h = 0;
  int nbtm = bottom.size();
  int bottom_h = bottom[nbtm - 1]->count();
  label_.Reshape(vector<int> {bottom_h});
  int* label_data = label_.mutable_gpu_data();
  const Dtype* bottom_data = bottom[nbtm - 1]->gpu_data();
  octree::generate_label_gpu<Dtype>(label_data, top_h, bottom_data, bottom_h, mask_);

  // deal with degenarated case
  if (top_h == 0) {
    top_h = 1;
    caffe_gpu_set(1, 0, label_data);
    LOG(INFO) << "Warning: split_num == 0 in layer " << this->layer_param_.name();
  }

  // copy data
  for (int i = 0; i < top.size(); ++i) {
    int channel = bottom[i]->shape(1);
    top[i]->Reshape(vector<int> {1, channel, top_h, 1});
    octree::pad_backward_gpu<Dtype>(top[i]->mutable_gpu_data(), top_h, channel,
        bottom[i]->gpu_data(), bottom_h, label_data);
  }
}

template <typename Dtype>
void OctreeMaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < top.size(); ++i) {
    if (!propagate_down[i]) continue;
    int channel = top[i]->shape(1);
    int top_h = top[i]->shape(2);
    int bottom_h = bottom[i]->shape(2);
    octree::pad_forward_gpu<Dtype>(bottom[i]->mutable_gpu_diff(), bottom_h, channel,
        top[i]->gpu_diff(), top_h, label_.gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeMaskLayer);
}