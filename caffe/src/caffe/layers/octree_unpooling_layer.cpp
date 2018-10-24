#include <vector>

#include "caffe/layers/octree_unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/octree.hpp"

namespace caffe {
template<typename Dtype>
void OctreeUnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
          << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();
  //curr_depth_ = Octree::get_curr_depth();
  //Octree::set_curr_depth(curr_depth_ + 1);

  buffer_ = Octree::get_workspace(Dtype(0));
}

template <typename Dtype>
void OctreeUnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // for the first time reshape
  if (top[0]->count() == 0) {
    vector<int> top_shape;
    top_shape = bottom[0]->shape();
    top_shape[2] = 8;
    top_shape[3] = 1;
    top[0]->Reshape(top_shape);
    return;
  }

  bool octree_in = bottom.size() == 3;
  Blob<Dtype>& the_octree = octree_in ? *bottom[2] : Octree::get_octree(Dtype(0));
  octree::set_octree_parser(octree_batch_, the_octree);

  // check
  int bottom_h = bottom[0]->shape(2);
  CHECK_EQ(bottom_h, octree_batch_.info().node_num(curr_depth_));

  // reshape buffer
  vector<int> buffer_shape = bottom[0]->shape();
  buffer_shape[2] = octree_batch_.info().node_num_nempty(curr_depth_);
  buffer_->Reshape(buffer_shape);
  CHECK_EQ(bottom[1]->shape(2), buffer_shape[2])
      << "The height of the mask is incorrect.";

  // reshape top blob
  vector<int> top_shape = bottom[0]->shape();
  top_shape[2] = octree_batch_.info().node_num(curr_depth_ + 1);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void OctreeUnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // de-padding
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int buffer_h = octree_batch_.info().node_num_nempty(curr_depth_);
  octree::pad_backward_cpu<Dtype>(buffer_->mutable_cpu_data(),
      buffer_h, channel, bottom[0]->cpu_data(), bottom_h,
      octree_batch_.children_cpu(curr_depth_));

  // unpooling
  const int* mask = (const int*)bottom[1]->cpu_data();
  const Dtype* btm_data = buffer_->cpu_data();
  int top_h = top[0]->shape(2);
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < buffer_h; ++h) {
      top_data[mask[h]] = btm_data[h];
    }

    // update pointer
    btm_data += buffer_h;
    mask += buffer_h;
    top_data += top_h;
  }
}

template <typename Dtype>
void OctreeUnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  // unpooling: backward
  int top_h = top[0]->shape(2);
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int buffer_h = octree_batch_.info().node_num_nempty(curr_depth_);
  Dtype* buffer_diff = buffer_->mutable_cpu_data();
  const int* mask = (const int*)bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < buffer_h; ++h) {
      buffer_diff[h] = top_diff[mask[h]];
    }

    // update pointer
    buffer_diff += buffer_h;
    mask += buffer_h;
    top_diff += top_h;
  }

  // pad: backward
  octree::pad_forward_cpu<Dtype>(bottom[0]->mutable_cpu_diff(),
      bottom_h, channel, buffer_->cpu_data(), buffer_h,
      octree_batch_.children_cpu(curr_depth_));

}

#ifdef CPU_ONLY
STUB_GPU(OctreeUnpoolingLayer);
#endif

INSTANTIATE_CLASS(OctreeUnpoolingLayer);
REGISTER_LAYER_CLASS(OctreeUnpooling);

}  // namespace caffe
