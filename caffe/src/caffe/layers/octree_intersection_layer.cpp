#include <vector>

#include "caffe/layers/octree_intersection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
OctreeIntersectionLayer<Dtype>::OctreeIntersectionLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {
  auto octree_param = this->layer_param_.octree_param();
  CHECK(octree_param.has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = octree_param.curr_depth();
  CHECK_GE(curr_depth_, 2)
      << "Error in " << this->layer_param_.name();
}

template <typename Dtype>
void OctreeIntersectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // a workaround for the first time reshape
  if (top[0]->count() == 0) {
    vector<int> top_shape{ 1 };
    top[0]->Reshape(top_shape);
    top[1]->Reshape(top_shape);
  }
}

template <typename Dtype>
void OctreeIntersectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void OctreeIntersectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(OctreeIntersectionLayer);
#endif

INSTANTIATE_CLASS(OctreeIntersectionLayer);
REGISTER_LAYER_CLASS(OctreeIntersection);

}  // namespace caffe
