#include "caffe/layers/octree_update_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void OctreeUpdateLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  auto octree_param = this->layer_param_.octree_param();
  CHECK(octree_param.has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = octree_param.curr_depth();

  CHECK_GT(octree_param.mask_size(), 0)
      << "Error in " << this->layer_param_.name()
      << ": Provide at least one mask value.";
  mask_ = octree_param.mask(0);
}

template <typename Dtype>
void OctreeUpdateLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void OctreeUpdateLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
  Dtype* octree_data = top[0]->mutable_cpu_data();
  octree_batch_.set_cpu(octree_data);

  int bottom_h = bottom[1]->count();
  int node_num = octree_batch_.info().node_num(curr_depth_);
  CHECK_EQ(node_num, bottom_h) << "Error in layer: " << this->layer_param_.name();

  // update children
  int split_num = 0;
  const Dtype* bottom_data = bottom[1]->cpu_data();
  int* children = octree_batch_.mutable_children_cpu(curr_depth_);
  octree::generate_label_cpu<Dtype>(children, split_num, bottom_data, bottom_h, mask_);

  // deal with degenarated case
  if (split_num == 0) {
    split_num = 1;
    children[0] = 0;
    LOG(INFO) << "Warning: split_num == 0 in layer " << this->layer_param_.name();
  }

  // update non-empty node number
  OctreeInfo* oct_info = reinterpret_cast<OctreeInfo*>(octree_data);
  oct_info->set_nempty(curr_depth_, split_num);
}

template <typename Dtype>
void OctreeUpdateLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
  // todo: whether does it need the gradient of bottom[0]?
}

#ifdef CPU_ONLY
STUB_GPU(OctreeUpdateLabelLayer);
#endif

INSTANTIATE_CLASS(OctreeUpdateLabelLayer);
REGISTER_LAYER_CLASS(OctreeUpdateLabel);

}  // namespace caffe
