#include "caffe/layers/octree_set_feature_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace caffe {

template <typename Dtype>
void OctreeSetFeatureLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  OctreeParser oct_parser;
  oct_parser.set_gpu(top[0]->mutable_gpu_data(), &oct_info_);

  // copy key and child
  int nnum = oct_parser_btm_.info().total_nnum();
  caffe_copy(nnum, oct_parser_btm_.key_gpu(0), oct_parser.mutable_key_gpu(0));
  caffe_copy(nnum, oct_parser_btm_.children_gpu(0), oct_parser.mutable_children_gpu(0));

  // copy data
  vector<int> flags(oct_info_.depth() + 1, 0);
  for (int i = 1; i < bottom.size(); ++i) {
    int depth = 0;
    int nnum = bottom[i]->shape(2);
    for (int d = 1; d <= oct_info_.depth(); ++d) {
      if (oct_info_.node_num(d) == nnum && flags[d] == 0) {
        flags[d] = 1;  depth = d;  break;
      }
    }
    CHECK_NE(depth, 0) << "Can not find the right octree layer";

    int num = bottom[i]->count();
    const Dtype* src = bottom[i]->gpu_data();
    float* des = oct_parser.mutable_feature_gpu(depth);
    thrust::transform(thrust::device, src, src + num, des, thrust::placeholders::_1);
  }
}

template <typename Dtype>
void OctreeSetFeatureLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeSetFeatureLayer);

}  // namespace caffe