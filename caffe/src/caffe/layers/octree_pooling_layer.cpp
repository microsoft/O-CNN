#include <vector>

#include "caffe/layers/octree_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/octree.hpp"

namespace caffe {
template<typename Dtype>
void OctreePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();
  //curr_depth_ = Octree::get_curr_depth();
  //Octree::set_curr_depth(curr_depth_ - 1);

  top_buffer_ = Octree::get_workspace(Dtype(0));
}

template <typename Dtype>
void OctreePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // check
  CHECK(bottom[0] != top[0]) << "In-place computation is not allowed";

  if (top[0]->count() == 0) {
    // a workaround for the first time reshape
    vector<int> top_shape = bottom[0]->shape();
    top_shape[2] = (curr_depth_ < 3) ?
        Octree::get_batchsize() * (1 << 3 * (curr_depth_ - 1)) : 8;
    top_shape[3] = 1;
    top[0]->Reshape(top_shape);
  } else {
    bool octree_in = bottom.size() == 2;
    Blob<Dtype>& the_octree = octree_in ? *bottom[1] : Octree::get_octree(Dtype(0));
    octree::set_octree_parser(octree_batch_, the_octree);

    // check
    int bottom_h = bottom[0]->shape(2);
    int nnum = octree_batch_.info().node_num(curr_depth_);
    if (nnum == 0) nnum = 1;
    CHECK_EQ(bottom_h, nnum) << "Error in " << this->layer_param_.name();

    // reshape the max-pooling index
    vector<int> idx_shape = bottom[0]->shape();
    idx_shape[2] = bottom_h >> 3;
    if (top.size() == 2) top[1]->Reshape(idx_shape);
    else max_idx_.Reshape(idx_shape);

    // the buffer contains the temporary results
    top_buffer_shape_ = bottom[0]->shape();
    top_buffer_shape_[2] = bottom_h >> 3;
    top_buffer_->Reshape(top_buffer_shape_);

    // reshape top blob
    vector<int> top_shape = bottom[0]->shape();
    top_shape[2] = octree_batch_.info().node_num(curr_depth_ - 1);
    top[0]->Reshape(top_shape);
  }
}

template <typename Dtype>
void OctreePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = top.size() == 2 ?
      (int*)top[1]->mutable_cpu_data() : max_idx_.mutable_cpu_data();
  Dtype* top_data = top_buffer_->mutable_cpu_data();
  const Dtype* btm_data = bottom[0]->cpu_data();

  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int top_h = bottom_h / 8;
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < top_h; ++h) {
      int hb = 8 * h;
      top_data[h] = btm_data[hb];
      mask[h] = hb;
      for (int idx = hb + 1; idx < hb + 8; ++idx) {
        if (btm_data[idx] > top_data[h]) {
          top_data[h] = btm_data[idx];
          mask[h] = idx;
        }
      }
    }

    // update pointer
    btm_data += bottom_h;
    top_data += top_h;
    mask += top_h;
  }

  // pad
  octree::pad_forward_cpu<Dtype>(top[0]->mutable_cpu_data(),
      top[0]->shape(2), top[0]->shape(1), top_buffer_->cpu_data(),
      top_buffer_shape_[2], octree_batch_.children_cpu(curr_depth_ - 1));
}

template <typename Dtype>
void OctreePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  // de-pad
  octree::pad_backward_cpu<Dtype>(top_buffer_->mutable_cpu_data(),
      top_buffer_shape_[2], top_buffer_shape_[1], top[0]->cpu_diff(),
      top[0]->shape(2), octree_batch_.children_cpu(curr_depth_ - 1));

  const int* mask = top.size() == 2 ?
      (const int*)top[1]->cpu_data() : max_idx_.cpu_data();
  const Dtype* top_diff = top_buffer_->cpu_data();
  Dtype* btm_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), btm_diff);

  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int top_h = bottom_h / 8;
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < top_h; ++h) {
      btm_diff[mask[h]] = top_diff[h];
    }

    // update pointer
    btm_diff += bottom_h;
    top_diff += top_h;
    mask += top_h;
  }
}

#ifdef CPU_ONLY
STUB_GPU(OctreePoolingLayer);
#endif

INSTANTIATE_CLASS(OctreePoolingLayer);
REGISTER_LAYER_CLASS(OctreePooling);
}  // namespace caffe
