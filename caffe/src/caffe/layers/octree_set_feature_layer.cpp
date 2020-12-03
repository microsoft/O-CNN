#include "caffe/layers/octree_set_feature_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void OctreeSetFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  is_adaptive_ = this->layer_param_.octree_param().is_adaptive();
  curr_depth_ = this->layer_param_.octree_param().curr_depth();
  adap_depth_ = this->layer_param_.octree_param().adapt_depth();
  CHECK_LE(adap_depth_, curr_depth_);

  int bottom_num = bottom.size();
  location_ = bottom_num > 2 ? -1 : curr_depth_;

  channel_ = bottom[1]->shape(1);
  for (int i = 2; i < bottom_num; ++i) {
    CHECK_EQ(bottom[i]->shape(1), channel_);
  }

  CHECK_NE(top[0], bottom[0]) << "The octree should not be shared";
}

template <typename Dtype>
void OctreeSetFeatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (top[0]->count() == 0) {
    top[0]->Reshape(vector<int> {1});
    return;
  }

  octree::set_octree_parser(oct_parser_btm_, *bottom[0]);
  oct_info_ = oct_parser_btm_.info();
  oct_info_.set_adaptive(is_adaptive_);
  oct_info_.set_adaptive_layer(adap_depth_);
  // Assume that the first 3 channels store normal, and the 4th channel is displacement.
  // TODO: add an interface in the caffe.proto in the future.
  oct_info_.set_node_dis(channel_ > 3);
  CHECK_EQ(oct_info_.depth(), curr_depth_);
  CHECK_EQ(oct_info_.channel(OctreeInfo::kChild), 1);
  CHECK_EQ(oct_info_.locations(OctreeInfo::kChild), -1);
  CHECK_EQ(oct_info_.channel(OctreeInfo::kKey), 1);
  CHECK_EQ(oct_info_.locations(OctreeInfo::kKey), -1);
  oct_info_.set_property(OctreeInfo::kNeigh, 0, 0);
  oct_info_.set_property(OctreeInfo::kFeature, channel_, location_);
  oct_info_.set_nnum_cum(0); // update the capacity of the octree
  oct_info_.set_ptr_dis();   // update the ptr for the Feature field

  int sz = oct_info_.sizeof_octree() / sizeof(Dtype) + 1;
  top[0]->Reshape(vector<int> {sz});
}

template <typename Dtype>
void OctreeSetFeatureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  OctreeParser oct_parser;
  oct_parser.set_cpu(top[0]->mutable_cpu_data(), &oct_info_);

  // copy key and child
  int nnum = oct_parser_btm_.info().total_nnum();
  caffe_copy(nnum, oct_parser_btm_.key_cpu(0), oct_parser.mutable_key_cpu(0));
  caffe_copy(nnum, oct_parser_btm_.children_cpu(0), oct_parser.mutable_children_cpu(0));

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

    const Dtype* src = bottom[i]->cpu_data();
    float* des = oct_parser.mutable_feature_cpu(depth);
    for (int j = 0; j < bottom[i]->count(); ++j) {
      des[j] = static_cast<float>(src[j]);
    }
  }
}

template <typename Dtype>
void OctreeSetFeatureLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

#ifdef CPU_ONLY
STUB_GPU(OctreeSetFeatureLayer);
#endif

INSTANTIATE_CLASS(OctreeSetFeatureLayer);
REGISTER_LAYER_CLASS(OctreeSetFeature);

}  // namespace caffe
