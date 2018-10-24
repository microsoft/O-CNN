#include <vector>
#include "caffe/layers/octree_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void OctreeMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  auto& octree_param = this->layer_param_.octree_param();
  CHECK_GT(octree_param.mask_size(), 0) << "Error in " << this->layer_param_.name()
      << ": Provide at least one mask value.";
  // currently only consider the first mask
  mask_ = octree_param.mask(0);

  // masks_.clear();  // todo: multi-mask
  //std::copy(masks_input.begin(), masks_input.end(), std::back_inserter(masks_));
}

template <typename Dtype>
void OctreeMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // a workaround for the first time reshape
  int nbtm = bottom.size();
  if (top[0]->count() == 0) {
    for (int i = 0; i < nbtm - 1; ++i) {
      vector<int> top_shape = bottom[i]->shape();
      top_shape[2] = 8;
      top[i]->Reshape(top_shape);
    }
    return;
  }

  int btm_h = bottom[nbtm - 1]->count();
  for (int i = 0; i < nbtm - 1; ++i) {
    vector<int> btm_shape = bottom[i]->shape();
    CHECK(btm_shape.size() == 4 && btm_shape[0] == 1 && btm_shape[2] == btm_h &&
          btm_shape[3] == 1) << "Error in " << this->layer_param_.name();
  }
  CHECK_EQ(nbtm, top.size() + 1) << "Error in " << this->layer_param_.name();
}

template <typename Dtype>
void OctreeMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // generate the map
  int top_h = 0;
  int nbtm = bottom.size();
  int bottom_h = bottom[nbtm - 1]->count();
  label_.Reshape(vector<int> {bottom_h});
  int* label_data = label_.mutable_cpu_data();
  const Dtype* bottom_data = bottom[nbtm - 1]->cpu_data();
  octree::generate_label_cpu<Dtype>(label_data, top_h, bottom_data, bottom_h, mask_);

  // deal with degenerated case
  if (top_h == 0) {
    top_h = 1;
    label_data[0] = 0;
    LOG(INFO) << "Warning: top_h == 0 in layer " << this->layer_param_.name();
  }

  // copy data
  for (int i = 0; i < top.size(); ++i) {
    int channel = bottom[i]->shape(1);
    top[i]->Reshape(vector<int> {1, channel, top_h, 1});
    octree::pad_backward_cpu<Dtype>(top[i]->mutable_cpu_data(), top_h, channel,
        bottom[i]->cpu_data(), bottom_h, label_data);
  }
}

template <typename Dtype>
void OctreeMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < top.size(); ++i) {
    if (!propagate_down[i]) continue;
    int channel = top[i]->shape(1);
    int top_h = top[i]->shape(2);
    int bottom_h = bottom[i]->shape(2);
    octree::pad_forward_cpu<Dtype>(bottom[i]->mutable_cpu_diff(), bottom_h, channel,
        top[i]->cpu_diff(), top_h, label_.cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU(OctreeMaskLayer);
#endif

INSTANTIATE_CLASS(OctreeMaskLayer);
REGISTER_LAYER_CLASS(OctreeMask);

}  // namespace caffe
