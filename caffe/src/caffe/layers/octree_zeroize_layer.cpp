#include "caffe/layers/octree_zeroize_layer.hpp"

namespace caffe {

template<typename Dtype>
void OctreeZeroizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size() + 1) << "Error in " << this->layer_param_.name();

  auto& octree_param = this->layer_param_.octree_param();
  CHECK_GT(octree_param.mask_size(), 0) << "Error in " << this->layer_param_.name()
      << ": Provide at least one mask value.";
  // currently only consider the first mask
  mask_ = octree_param.mask(0);

}

template <typename Dtype>
void OctreeZeroizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int blob_num = top.size();
  int bottom_h = bottom[blob_num]->count();
  for (int i = 0; i < blob_num; ++i) {
    top[i]->ReshapeLike(*bottom[i]);
    CHECK_EQ(top[i]->shape(2), bottom_h);
  }
}

template <typename Dtype>
void OctreeZeroizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int blob_num = top.size();
  int height = bottom[blob_num]->count();
  const Dtype* label_data = bottom[blob_num]->cpu_data();
  for (int i = 0; i < blob_num; ++i) {
    int channel = bottom[i]->shape(1);
    Dtype* top_data = top[i]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < height; ++h) {
        top_data[c * height + h] = static_cast<int>(label_data[h]) == mask_ ?
            Dtype(0) : bottom_data[c * height + h];
      }
    }
  }
}

template <typename Dtype>
void OctreeZeroizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int blob_num = top.size();
  int height = bottom[blob_num]->count();
  const Dtype* label_data = bottom[blob_num]->cpu_data();
  for (int i = 0; i < blob_num; ++i) {
    if (!propagate_down[i]) continue;
    int channel = bottom[i]->shape(1);
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    const Dtype* top_diff = top[i]->cpu_diff();
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < height; ++h) {
        bottom_diff[c * height + h] = static_cast<int>(label_data[h]) == mask_ ?
            Dtype(0) : top_diff[c * height + h];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(OctreeZeroizeLayer);
#endif

INSTANTIATE_CLASS(OctreeZeroizeLayer);
REGISTER_LAYER_CLASS(OctreeZeroize);

}  // namespace caffe
