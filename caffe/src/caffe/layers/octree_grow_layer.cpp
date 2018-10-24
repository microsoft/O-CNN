#include <vector>
#include "caffe/layers/octree_grow_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
int OctreeGrowLayer<Dtype>::node_num_capacity_ = 1;

template <typename Dtype>
void OctreeGrowLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();
  CHECK_GE(curr_depth_,  0);

  full_octree_ = this->layer_param_.octree_param().full_octree();
  if (bottom.size() == 0) { 
    CHECK(full_octree_); 
  } else {
    CHECK_NE(bottom[0], top[0])
        << " Error in: " << this->layer_param_.name() << ": "
        << "The bottom and top blob should not be in-place";
  }
  if (full_octree_ && bottom.size() == 0) {
    CHECK(this->layer_param_.octree_param().has_batch_size())
        << "Error in " << this->layer_param_.name() << ": "
        << "The batch_size must be set.";
    batch_size_ = this->layer_param_.octree_param().batch_size();
    CHECK_GE(batch_size_, 1);
  }

  // full octree. We generate the full octree only once
  full_octree_init_ = false;
}

template <typename Dtype>
void OctreeGrowLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // a work around for the first time reshape
  if (top[0]->count() == 0) {
    top[0]->Reshape(vector<int> { 1 });
    return;
  }

  // get node number in current octree depth
  if (bottom.size() == 0) {
    node_num_ = batch_size_ * (1 << curr_depth_ * 3);
  } else {
    octree::set_octree_parser(oct_parser_btm_, *bottom[0]);

    node_num_ = oct_parser_btm_.info().node_num_nempty(curr_depth_ - 1) << 3;
    batch_size_ = oct_parser_btm_.info().batch_size();

    // copy the octree information
    oct_info_ = oct_parser_btm_.info();
  }

  // update the octree info
  update_octreeinfo();

  // reshape the top blob
  if (oct_info_.total_nnum_capacity() > node_num_capacity_) {
    // reserve enough memory
    node_num_capacity_ = oct_info_.total_nnum_capacity() * 1.5;
  }
  int sz = oct_info_.sizeof_octree() / sizeof(Dtype) + 1;
  if (top[0]->count() != sz) {
    // Normally, top[0] shares data with bottom[0] when bottom.size() != 0.
    // After calling top[0]->Reshape(), the sharing relationship between top[0]
    // and bottom[0] is broken, because the Reshape() reset the blob's data_
    // via data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype))).
    // So full_octree_init_ is set to false, and in the next iteration the
    // bottom[0] is Reshaped to vector<int> {blob_capacity_} and re-initialized
    // The sharing relationship between top[0] and bottom[0] is re-built.
    top[0]->Reshape(vector<int> {sz});
    full_octree_init_ = false;
  }
}

template <typename Dtype>
void OctreeGrowLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  OctreeParser oct_parser;
  if (bottom.size() != 0) {
    if (bottom[0]->count() != top[0]->count()) {
      // The size are not equal, i.e. the top[0] is reshaped and the sharing
      // relationship is broken, so copy the data from bottom[0] to top[0]
      int nnum = oct_parser_btm_.info().total_nnum();
      int nnum_ngh = nnum * oct_parser_btm_.info().channel(OctreeInfo::kNeigh);
      oct_parser.set_cpu(top[0]->mutable_cpu_data(), &oct_info_);
      caffe_copy(nnum, oct_parser_btm_.key_cpu(0), oct_parser.mutable_key_cpu(0));
      caffe_copy(nnum, oct_parser_btm_.children_cpu(0), oct_parser.mutable_children_cpu(0));
      caffe_copy(nnum_ngh, oct_parser_btm_.neighbor_cpu(0), oct_parser.mutable_neighbor_cpu(0));
    } else {
      // sharing data between top[0] and bottom[0]
      top[0]->ShareData(*bottom[0]);
    }
  }

  if (full_octree_) {
    // for full octree, just run the forward pass once
    if (full_octree_init_) { return; }
    full_octree_init_ = true;
    oct_parser.set_cpu(top[0]->mutable_cpu_data(), &oct_info_);

    octree::calc_neigh_cpu(oct_parser.mutable_neighbor_cpu(curr_depth_),
        curr_depth_, batch_size_);

    octree::generate_key_cpu(oct_parser.mutable_key_cpu(curr_depth_),
        curr_depth_, batch_size_);

    int* children = oct_parser.mutable_children_cpu(curr_depth_);
    for (int i = 0; i < node_num_; ++i) children[i] = i;
  } else {
    oct_parser.set_cpu(top[0]->mutable_cpu_data(), &oct_info_);

    const int* label_ptr = oct_parser.children_cpu(curr_depth_ - 1);
    octree::calc_neigh_cpu(oct_parser.mutable_neighbor_cpu(curr_depth_),
        oct_parser.neighbor_cpu(curr_depth_ - 1), label_ptr,
        oct_parser.info().node_num(curr_depth_ - 1));

    octree::generate_key_cpu(oct_parser.mutable_key_cpu(curr_depth_),
        oct_parser.key_cpu(curr_depth_ - 1), label_ptr,
        oct_parser.info().node_num(curr_depth_ - 1));

    int* children = oct_parser.mutable_children_cpu(curr_depth_);
    for (int i = 0; i < node_num_; ++i) children[i] = i;
  }
}

template<typename Dtype>
void OctreeGrowLayer<Dtype>::update_octreeinfo() {
  oct_info_.set_batch_size(batch_size_);
  oct_info_.set_depth(curr_depth_);
  if (full_octree_) {
    oct_info_.set_full_layer(curr_depth_);
    oct_info_.set_adaptive(false);
    oct_info_.set_key2xyz(true);   //!!! todo: key2xyz = false
    oct_info_.set_property(OctreeInfo::kKey, 1, -1);
    oct_info_.set_property(OctreeInfo::kChild, 1, -1);
    oct_info_.set_property(OctreeInfo::kNeigh, 8, -1);
  }
  float width = 1 << curr_depth_;
  float bbmin[] = { 0, 0, 0 };
  float bbmax[] = { width, width, width };
  oct_info_.set_bbox(bbmin, bbmax);
  oct_info_.set_nnum(curr_depth_, node_num_);
  // Just set the non-empty node number as node_num_,
  // it needs to be updated by the spliting label
  oct_info_.set_nempty(curr_depth_, node_num_);
  oct_info_.set_nnum_cum(node_num_capacity_);
  oct_info_.set_ptr_dis();
}

template <typename Dtype>
void OctreeGrowLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // do nothing
}

#ifdef CPU_ONLY
STUB_GPU(OctreeGrowLayer);
#endif

INSTANTIATE_CLASS(OctreeGrowLayer);
REGISTER_LAYER_CLASS(OctreeGrow);

}  // namespace caffe
