#include "caffe/layers/octree_property_layer.hpp"

#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace caffe {
template <typename Dtype>
void OctreePropertyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < ptypes_.size(); ++i) {
    const char* ptr = octree_batch_.ptr_gpu(ptypes_[i], curr_depth_);
    CHECK(ptr != nullptr) << "The octree property does not exist: " << ptypes_[i];
    if (sizeof(Dtype) == 8 && IsDataProperty(ptypes_[i])) {
      int num = top[i]->count();
      Dtype* des = top[i]->mutable_gpu_data();
      const float* src = reinterpret_cast<const float*>(ptr);
      thrust::transform(thrust::device, src, src + num, des, thrust::placeholders::_1);
    } else {
      top[i]->set_gpu_data(reinterpret_cast<Dtype*>(const_cast<char*>(ptr)));
    }
  }
}

template <typename Dtype>
void OctreePropertyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreePropertyLayer);
}