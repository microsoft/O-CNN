#include "caffe/layers/octree_grow_layer.hpp"
#include "caffe/util/gpu_util.cuh"

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

namespace caffe {

template <typename Dtype>
void OctreeGrowLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  OctreeParser oct_parser;
  if (bottom.size() != 0) {
    if (bottom[0]->count() != top[0]->count()) {
      // The size are not equal, i.e. the top[0] is reshaped and the sharing
      // relationship is broken, so copy the data from bottom[0] to top[0]
      int nnum = oct_parser_btm_.info().total_nnum();
      int nnum_ngh = nnum * oct_parser_btm_.info().channel(OctreeInfo::kNeigh);
      oct_parser.set_gpu(top[0]->mutable_gpu_data(), &oct_info_);
      caffe_copy(nnum, oct_parser_btm_.key_gpu(0), oct_parser.mutable_key_gpu(0));
      caffe_copy(nnum, oct_parser_btm_.children_gpu(0), oct_parser.mutable_children_gpu(0));
      caffe_copy(nnum_ngh, oct_parser_btm_.neighbor_gpu(0), oct_parser.mutable_neighbor_gpu(0));
    } else {
      // sharing data between top[0] and bottom[0]
      top[0]->ShareData(*bottom[0]);
    }
  }

  if (full_octree_) {
    // for full octree, just run the forward pass once
    if (full_octree_init_) { return; }
    full_octree_init_ = true;
    oct_parser.set_gpu(top[0]->mutable_gpu_data(), &oct_info_);

    octree::calc_neigh_gpu(oct_parser.mutable_neighbor_gpu(curr_depth_),
        curr_depth_, batch_size_);

    octree::generate_key_gpu(oct_parser.mutable_key_gpu(curr_depth_),
        curr_depth_, batch_size_);

    int* children = oct_parser.mutable_children_gpu(curr_depth_);
    thrust::sequence(thrust::device, children, children + node_num_);
  } else {
    oct_parser.set_gpu(top[0]->mutable_gpu_data(), &oct_info_);

    const int* label_ptr = oct_parser.children_gpu(curr_depth_ - 1);
    octree::calc_neigh_gpu(oct_parser.mutable_neighbor_gpu(curr_depth_),
        oct_parser.neighbor_gpu(curr_depth_ - 1), label_ptr,
        oct_parser.info().node_num(curr_depth_ - 1));

    octree::generate_key_gpu(oct_parser.mutable_key_gpu(curr_depth_),
        oct_parser.key_gpu(curr_depth_ - 1), label_ptr,
        oct_parser.info().node_num(curr_depth_ - 1));

    int* children = oct_parser.mutable_children_gpu(curr_depth_);
    thrust::sequence(thrust::device, children, children + node_num_);
  }
}

template <typename Dtype>
void OctreeGrowLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeGrowLayer);

}  // namespace caffe