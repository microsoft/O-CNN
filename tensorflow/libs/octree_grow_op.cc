#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "octree_nn.h"
#include "octree_parser.h"

namespace tensorflow {

REGISTER_OP("OctreeGrow")
    .Input("in_octree: int8")
    .Attr("target_depth: int")
    .Attr("full_octree: bool = false")
    .Output("out_octree: int8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(Octree grow operator.)doc");

class OctreeGrowOp : public OpKernel {
 public:
  explicit OctreeGrowOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("target_depth", &target_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("full_octree",  &full_octree_));
  }

  void Compute(OpKernelContext* context) override {
    // in octree
    OctreeParser octree_in;
    octree_in.set_gpu(context->input(0).flat<int8>().data());

    // out info
    batch_size_ = octree_in.info().batch_size();
    node_num_ = octree_in.info().node_num_nempty(target_depth_ - 1) << 3;
    OctreeInfo oct_info_;
    oct_info_ = octree_in.info();
    update_octreeinfo(oct_info_);

    // out octree
    Tensor* tensor_out = nullptr;
    TensorShape shape_out({oct_info_.sizeof_octree()});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_out, &tensor_out));
    auto* ptr_out = tensor_out->flat<int8>().data();
    memset_gpu(tensor_out->NumElements(), char(0), (char*)ptr_out);
    // cudaMemset(ptr_out, 0, tensor_out->NumElements());

    // copy octree
    OctreeParser octree_out;
    octree_out.set_gpu(ptr_out, &oct_info_);
    copy_octree_gpu(octree_out, octree_in);

    // grow octree
    if (full_octree_) {
      calc_neigh_gpu(octree_out.mutable_neighbor_gpu(target_depth_),
                     target_depth_, batch_size_);
      generate_key_gpu(octree_out.mutable_key_gpu(target_depth_), 
                       target_depth_, batch_size_);
      sequence_gpu(octree_out.mutable_children_gpu(target_depth_), node_num_);
    } else {
      Tensor displacement, parent;
      init_neigh_ptrs(context, parent, displacement);
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
  }

 private:
  void update_octreeinfo(OctreeInfo& oct_info_) {
    oct_info_.set_depth(target_depth_);
    if (full_octree_) {
      oct_info_.set_full_layer(target_depth_);
    }
    float width = 1 << target_depth_;
    float bbmin[] = {0, 0, 0};
    float bbmax[] = {width, width, width};
    oct_info_.set_bbox(bbmin, bbmax);
    oct_info_.set_nnum(target_depth_, node_num_);
    // Just set the non-empty node number as node_num_,
    // it needs to be updated by the new node-splitting label
    oct_info_.set_nempty(target_depth_, node_num_);
    oct_info_.set_nnum_cum();
    oct_info_.set_ptr_dis();
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
  void init_neigh_ptrs(OpKernelContext* ctx, Tensor& parent, Tensor& dis) {
    const vector<int>& dis_cpu = NeighHelper::Get().get_dis_array();
    TensorShape dshape({(long long int)dis_cpu.size()});
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, dshape, &dis));
    ptr_dis_ = dis.flat<int>().data();
    memcpy_gpu(dis_cpu.size(), dis_cpu.data(), ptr_dis_);

    const vector<int>& parent_cpu = NeighHelper::Get().get_parent_array();
    TensorShape pshape({(long long int)parent_cpu.size()});
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, pshape, &parent));
    ptr_parent_ = parent.flat<int>().data();
    memcpy_gpu(parent_cpu.size(), parent_cpu.data(), ptr_parent_);
  }

 private:
  int batch_size_;
  int target_depth_;
  int node_num_;
  bool full_octree_;
  int* ptr_parent_;
  int* ptr_dis_;
};

REGISTER_KERNEL_BUILDER(Name("OctreeGrow").Device(DEVICE_GPU), OctreeGrowOp);

}  // namespace tensorflow
