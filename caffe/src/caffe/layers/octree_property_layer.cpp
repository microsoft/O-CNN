#include "caffe/layers/octree_property_layer.hpp"

namespace caffe {

template<typename Dtype>
void OctreePropertyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const auto& octree_param = this->layer_param_.octree_param();
  CHECK(octree_param.has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = octree_param.curr_depth();
  signal_channel_ = octree_param.signal_channel();

  content_flag_ = 3;
  if (octree_param.has_content_flag()) {
    content_flag_ = octree::content_flag(octree_param.content_flag());
  }

  ptypes_.clear();
  for (int i = 0; i < OctreeInfo::kPTypeNum; ++i) {
    OctreeInfo::PropType p = static_cast<OctreeInfo::PropType>(1 << i);
    if (0 != (content_flag_ & p)) ptypes_.push_back(p);
  }

  int blob_num = ptypes_.size();
  CHECK_EQ(blob_num, top.size())
      << "Error in " << this->layer_param_.name()
      << ": the content_flag and top blob size should be consistent!";
}

template <typename Dtype>
void OctreePropertyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // a workaround for the first time reshape
  if (top[0]->count() == 0) {
    vector<int> top_shape{ 1, signal_channel_, 8, 1 };
    for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
    }
    return;
  }

  // set octree parser
  bool octree_in = bottom.size() == 1;
  Blob<Dtype>& the_octree = octree_in ? *bottom[0] : Octree::get_octree(Dtype(0));
  octree::set_octree_parser(octree_batch_, the_octree);

  int nnum = octree_batch_.info().node_num(curr_depth_);
  if (nnum == 0) {
    LOG(INFO) << "Warning in " << this->layer_param_.name()
        << ": This octree layer is empty";
  }
  for (int i = 0; i < top.size(); ++i) {
    OctreeInfo::PropType ptype = ptypes_[i];
    int channel = octree_batch_.info().channel(ptype);
    // Requir that all the signal channel should be the same
    CHECK_EQ(channel, signal_channel_)
        << "The signal_channel_ is not consistent: " << ptypes_[i];
    int height = nnum;
    if (!IsDataProperty(ptype) && sizeof(Dtype) == 8) {
      height = (nnum + 1) / 2; // ceil
    }
    top[i]->Reshape(vector<int> {1, channel, height, 1});
  }
}

template <typename Dtype>
void OctreePropertyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < ptypes_.size(); ++i) {
    const char* ptr = octree_batch_.ptr_cpu(ptypes_[i], curr_depth_);
    CHECK(ptr != nullptr) << "The octree property does not exist: " << ptypes_[i];
    if (sizeof(Dtype) == 8 && IsDataProperty(ptypes_[i])) {
      Dtype* des = top[i]->mutable_cpu_data();
      const float* src = reinterpret_cast<const float*>(ptr);
      for (int j = 0; j < top[i]->count(); ++j) {
        des[j] = static_cast<Dtype>(src[j]);
      }
    } else {
      top[i]->set_cpu_data(reinterpret_cast<Dtype*>(const_cast<char*>(ptr)));
    }
  }
}

template <typename Dtype>
void OctreePropertyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

template <typename Dtype>
bool OctreePropertyLayer<Dtype>::IsDataProperty(OctreeInfo::PropType ptype) {
  return  ptype == OctreeInfo::kFeature || ptype == OctreeInfo::kLabel ||
      ptype == OctreeInfo::kSplit;
}

#ifdef CPU_ONLY
STUB_GPU(OctreePropertyLayer);
#endif

INSTANTIATE_CLASS(OctreePropertyLayer);
REGISTER_LAYER_CLASS(OctreeProperty);

}  // namespace caffe
