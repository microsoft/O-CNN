#include <vector>

#include "caffe/layers/octree_tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void OctreeTileLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();

  tile_depth_ = curr_depth_ + 1;
  if (this->layer_param_.octree_param().has_tile_depth()) {
    tile_depth_ = this->layer_param_.octree_param().tile_depth();
    CHECK_GT(tile_depth_, curr_depth_)
        << "Error in " << this->layer_param_.name() << ": "
        << "The tile_depth should be smaller than curr_depth.";
  }

  mask0_.reset(new Blob<int>(vector<int> {1}));
  mask1_.reset(new Blob<int>(vector<int> {1}));
}

template <typename Dtype>
void OctreeTileLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // a workaround for the first time reshape
  if (top[0]->count() == 0) {
    vector<int> top_shape = bottom[0]->shape();
    top_shape[2] = 8;
    top_shape[3] = 1;
    top[0]->Reshape(top_shape);
    return;
  }

  // get the octree
  bool octree_in = bottom.size() == 2;
  Blob<Dtype>& the_octree = octree_in ? *bottom[1] : Octree::get_octree(Dtype(0));
  octree::set_octree_parser(octree_batch_, the_octree);

  // check
  int bottom_h = bottom[0]->shape(2);
  CHECK_EQ(bottom_h, octree_batch_.info().node_num(curr_depth_))
      << "The node number is inconsistent in layer: "
      << this->layer_param_.name();
  CHECK_LE(tile_depth_, octree_batch_.info().depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The tile_depth should be smaller than octree depth.";

  // top_shape
  int top_h = octree_batch_.info().node_num(tile_depth_);
  vector<int> top_shape = bottom[0]->shape();
  top_shape[2] = top_h;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void OctreeTileLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // generate the copy mask
  int nnum = octree_batch_.info().node_num(curr_depth_);
  mask0_->Reshape(vector<int> {nnum});
  int* mask0_data = mask0_->mutable_cpu_data();
  for (int i = 0; i < nnum; ++i) mask0_data[i] = i;

  for (int d = curr_depth_; d < tile_depth_; ++d) {
    const int* children = octree_batch_.children_cpu(d);
    const int* mask0_data = mask0_->cpu_data();
    int nnum = octree_batch_.info().node_num(d);
    mask1_->Reshape(vector<int> { octree_batch_.info().node_num(d + 1) });
    int* mask1_data = mask1_->mutable_cpu_data();
    for (int i = 0; i < nnum; ++i) {
      int t = children[i];
      if (t == -1) continue;
      int t8 = t << 3;
      for (int j = 0; j < 8; ++j) {
        mask1_data[t8 + j] = mask0_data[i];
      }
    }

    mask0_.swap(mask1_);
  }

  // copy according to mask0_
  int top_h = top[0]->shape(2);
  int channel = top[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  const int* mask_data = mask0_->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < top_h; ++h) {
      top_data[c * top_h + h] = bottom_data[c * bottom_h + mask_data[h]];
    }
  }
}

template <typename Dtype>
void OctreeTileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int top_h = top[0]->shape(2);
  int channel = top[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  const int* mask_data = mask0_->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* btm_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), btm_diff);
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < top_h; ++h) {
      btm_diff[c * bottom_h + mask_data[h]] += top_diff[c * top_h + h];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(OctreeTileLayer);
#endif

INSTANTIATE_CLASS(OctreeTileLayer);
REGISTER_LAYER_CLASS(OctreeTile);

}  // namespace caffe
