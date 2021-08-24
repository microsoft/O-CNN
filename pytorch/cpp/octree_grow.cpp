#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

namespace {

class OctreeGrowOp {
 public:
  explicit OctreeGrowOp(int target_depth, bool full_octree)
      : target_depth_(target_depth), full_octree_(full_octree) {}

  Tensor compute(Tensor tensor_in) {
    // in octree
    OctreeParser octree_in;
    octree_in.set_gpu(tensor_in.data_ptr<uint8_t>());

    // out info
    batch_size_ = octree_in.info().batch_size();
    node_num_ = octree_in.info().node_num_nempty(target_depth_ - 1) << 3;
    OctreeInfo oct_info = octree_in.info();
    update_octreeinfo(oct_info);

    // out octree
    torch::TensorOptions options = tensor_in.options();
    Tensor tensor_out = torch::zeros({oct_info.sizeof_octree()}, options);

    // copy octree
    OctreeParser octree_out;
    octree_out.set_gpu(tensor_out.data_ptr<uint8_t>(), &oct_info);
    copy_octree_gpu(octree_out, octree_in);

    // grow octree
    if (full_octree_) {
      calc_neigh_gpu(octree_out.mutable_neighbor_gpu(target_depth_),
                     target_depth_, batch_size_);
      generate_key_gpu(octree_out.mutable_key_gpu(target_depth_), target_depth_,
                       batch_size_);
      sequence_gpu(octree_out.mutable_children_gpu(target_depth_), node_num_);
    } else {
      vector<Tensor> tmp = init_neigh_ptrs(options);
      const int* label_ptr = octree_out.children_gpu(target_depth_ - 1);
      calc_neigh_gpu(octree_out.mutable_neighbor_gpu(target_depth_),
                     octree_out.neighbor_gpu(target_depth_ - 1), label_ptr,
                     octree_out.info().node_num(target_depth_ - 1), ptr_parent_,
                     ptr_dis_);
      generate_key_gpu(octree_out.mutable_key_gpu(target_depth_),
                       octree_out.key_gpu(target_depth_ - 1), label_ptr,
                       octree_out.info().node_num(target_depth_ - 1));
      sequence_gpu(octree_out.mutable_children_gpu(target_depth_), node_num_);
    }

    return tensor_out;
  }

 private:
  void update_octreeinfo(OctreeInfo& oct_info) {
    oct_info.set_depth(target_depth_);
    if (full_octree_) {
      oct_info.set_full_layer(target_depth_);
    }
    float width = 1 << target_depth_;
    float bbmin[] = {0, 0, 0};
    float bbmax[] = {width, width, width};
    oct_info.set_bbox(bbmin, bbmax);
    oct_info.set_nnum(target_depth_, node_num_);
    // Just set the non-empty node number as node_num_,
    // it needs to be updated by the new node-splitting label
    oct_info.set_nempty(target_depth_, node_num_);
    oct_info.set_nnum_cum();
    oct_info.set_ptr_dis();
  }

  void copy_octree_gpu(OctreeParser& octree_o, const OctreeParser& octree_i) {
    int node_num_cum = octree_i.info().node_num_cum(target_depth_);
    int num = node_num_cum * octree_i.info().channel(OctreeInfo::kKey);
    memcpy_gpu(num, octree_i.key_gpu(0), octree_o.mutable_key_gpu(0));

    num = node_num_cum * octree_i.info().channel(OctreeInfo::kChild);
    memcpy_gpu(num, octree_i.children_gpu(0), octree_o.mutable_children_gpu(0));

    num = node_num_cum * octree_i.info().channel(OctreeInfo::kNeigh);
    memcpy_gpu(num, octree_i.neighbor_gpu(0), octree_o.mutable_neighbor_gpu(0));

    num = node_num_cum * octree_i.info().channel(OctreeInfo::kFeature);
    memcpy_gpu(num, octree_i.feature_gpu(0), octree_o.mutable_feature_gpu(0));
  }

  vector<Tensor> init_neigh_ptrs(torch::TensorOptions options) {
    const vector<int>& dis_cpu = NeighHelper::Get().get_dis_array();
    int count = dis_cpu.size();
    Tensor dis_gpu = torch::zeros({count}, options.dtype(torch::kInt32));
    ptr_dis_ = dis_gpu.data_ptr<int>();
    memcpy_gpu(count, dis_cpu.data(), ptr_dis_);

    const vector<int>& parent_cpu = NeighHelper::Get().get_parent_array();
    count = parent_cpu.size();
    Tensor parent_gpu = torch::zeros({count}, options.dtype(torch::kInt32));
    ptr_parent_ = parent_gpu.data_ptr<int>();
    memcpy_gpu(count, parent_cpu.data(), ptr_parent_);

    return vector<Tensor>{dis_gpu, parent_gpu};
  }

 private:
  int batch_size_;
  int target_depth_;
  int node_num_;
  bool full_octree_;
  int* ptr_parent_;
  int* ptr_dis_;
};

}  // anonymous namespace

// API implementation
Tensor octree_grow(Tensor octree_in, int target_depth, bool full_octree) {
  OctreeGrowOp op(target_depth, full_octree);
  return op.compute(octree_in);
}

Tensor octree_new(int batch_size, int channel, bool node_dis, int adaptive_layer) {
  CHECK_GE(batch_size, 1);
  int node_num = batch_size;
  int depth = 0;

  // octree info
  OctreeInfo oct_info;
  oct_info.set_batch_size(batch_size);
  oct_info.set_depth(depth);
  oct_info.set_full_layer(depth);
  oct_info.set_node_dis(node_dis);
  if (adaptive_layer > 1) {
    oct_info.set_adaptive(true);
    oct_info.set_adaptive_layer(adaptive_layer);
  } else {
    oct_info.set_adaptive(false);
  }
  oct_info.set_key2xyz(true);  // !!! NOTE: set_key2xyz with true
  oct_info.set_property(OctreeInfo::kKey, 1, -1);
  oct_info.set_property(OctreeInfo::kChild, 1, -1);
  oct_info.set_property(OctreeInfo::kNeigh, 8, -1);
  oct_info.set_property(OctreeInfo::kFeature, channel, -1);
  float bbmin[] = {0, 0, 0};
  float bbmax[] = {2, 2, 2};
  oct_info.set_bbox(bbmin, bbmax);
  oct_info.set_nnum(depth, node_num);
  oct_info.set_nnum_cum();
  oct_info.set_nempty(depth, node_num);
  oct_info.set_ptr_dis();

  // init output tensor
  Tensor tensor_out = torch::zeros({oct_info.sizeof_octree()}, torch::kUInt8);

  // set octree, skip the propoerties neigh and feature
  OctreeParser octree_out;
  octree_out.set_cpu(tensor_out.data_ptr<uint8_t>(), &oct_info);
  sequence_cpu(octree_out.mutable_key_cpu(depth), node_num);  // !!! NOTE: inconsitent with L140
  sequence_cpu(octree_out.mutable_children_cpu(depth), node_num);

  return tensor_out.cuda();
}

Tensor octree_update(Tensor octree_in, Tensor label_in, int depth, int split) {
  Tensor tensor_out = octree_in.clone();
  uint8_t* out_ptr = tensor_out.data_ptr<uint8_t>();
  OctreeParser octree_;
  octree_.set_gpu(out_ptr);
  int node_num = octree_.info().node_num(depth);

  label_in = label_in.contiguous();
  int* label_ptr = label_in.data_ptr<int>();
  CHECK_EQ(node_num, label_in.numel());

  // update children
  int split_num = 0;  // non-empty node number
  int* children = octree_.mutable_children_gpu(depth);
  generate_label_gpu(children, split_num, label_ptr, node_num, split);

  // deal with degenatated case
  if (split_num == 0) {
    split_num = 1;
    memset_gpu(1, 0, children);
    LOG(INFO) << "Warning: split_num == 0 in octree update layer.";
  }

  octree_.mutable_info().set_nempty(depth, split_num);
  memcpy_gpu(sizeof(OctreeInfo), (const char*)&octree_.info(), (char*)out_ptr);
  return tensor_out;
}
