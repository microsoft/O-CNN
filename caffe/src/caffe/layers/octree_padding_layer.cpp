#include <vector>
#include "caffe/layers/octree_padding_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template<typename Dtype>
void OctreePaddingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
          << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();
  //curr_depth_ = Octree::get_curr_depth();
}

template <typename Dtype>
void OctreePaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (top[0]->count() == 0) {
    // a workaround for the first time reshape
    vector<int> top_shape = bottom[0]->shape();
    top_shape[2] = 8;
    top[0]->Reshape(top_shape);
  } else {
    bool octree_in = bottom.size() == 2;
    Blob<Dtype>& the_octree = octree_in ? *bottom[1] : Octree::get_octree(Dtype(0));
    octree::set_octree_parser(octree_batch_, the_octree);

    int bottom_h = bottom[0]->shape(2);
    int top_h = octree_batch_.info().node_num(curr_depth_);
    if (top_h == bottom_h) {
      LOG(INFO) << "The layer " << this->layer_param_.name() << "is redundant.";
    } else {
      CHECK_EQ(bottom_h, octree_batch_.info().node_num_nempty(curr_depth_));
    }

    vector<int>top_shape = bottom[0]->shape();
    top_shape[2] = top_h;
    top[0]->Reshape(top_shape);
  }
}

template <typename Dtype>
void OctreePaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int channel = top[0]->shape(1);
  int top_h = top[0]->shape(2);
  int bottom_h = bottom[0]->shape(2);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (top_h != bottom_h) {
    octree::pad_forward_cpu<Dtype>(top_data, top_h, channel,
        bottom_data, bottom_h, octree_batch_.children_cpu(curr_depth_));
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void OctreePaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int channel = top[0]->shape(1);
  int top_h = top[0]->shape(2);
  int bottom_h = bottom[0]->shape(2);
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (top_h != bottom_h) {
    octree::pad_backward_cpu<Dtype>(bottom_diff, bottom_h, channel,
        top_diff, top_h, octree_batch_.children_cpu(curr_depth_));
  } else {
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(OctreePaddingLayer);
#endif

INSTANTIATE_CLASS(OctreePaddingLayer);
REGISTER_LAYER_CLASS(OctreePadding);

}  // namespace caffe
