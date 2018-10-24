#include "caffe/layers/octree_update_label_layer.hpp"
#include "caffe/util/gpu_util.cuh"

#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/replace.h>

namespace caffe {

template <typename Dtype>
void OctreeUpdateLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
  Dtype* octree_data = top[0]->mutable_gpu_data();
  OctreeInfo oct_info;
  CUDA_CHECK(cudaMemcpy(&oct_info, octree_data, sizeof(OctreeInfo), cudaMemcpyDeviceToHost));
  octree_batch_.set_gpu(octree_data, &oct_info);

  int bottom_h = bottom[1]->count();
  int node_num = octree_batch_.info().node_num(curr_depth_);
  CHECK_EQ(node_num, bottom_h);

  // update children
  int split_num = 0;
  const Dtype* bottom_data = bottom[1]->gpu_data();
  int* children = octree_batch_.mutable_children_gpu(curr_depth_);
  octree::generate_label_gpu<Dtype>(children, split_num, bottom_data, bottom_h, mask_);

  // deal with degenarated case
  if (split_num == 0) {
    split_num = 1;
    caffe_gpu_set(1, 0, children);
    LOG(INFO) << "Warning: split_num == 0 in layer " << this->layer_param_.name();
  }

  // update non-empty node number
  oct_info.set_nempty(curr_depth_, split_num);
  CUDA_CHECK(cudaMemcpy(octree_data, &oct_info, sizeof(OctreeInfo), cudaMemcpyHostToDevice));
}

template <typename Dtype>
void OctreeUpdateLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
  // todo: whether does it need the gradient of bottom[0]?
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeUpdateLabelLayer);

}  // namespace caffe 