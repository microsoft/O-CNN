#include "octree_nn.h"
#include "octree_parser.h"

#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("OctreeUpdate")
    .Input("in_octree: int8")
    .Input("in_label: int32")
    .Attr("depth: int")
    .Attr("mask: int")
    .Output("out_octree: int8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Octree update operator.)doc");


class OctreeUpdateOp : public OpKernel {
 public:
  explicit OctreeUpdateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &curr_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("mask", &mask_));
  }

  void Compute(OpKernelContext* context) override {
    // in octree
    const Tensor& in_octree = context->input(0);
    auto in_ptr = in_octree.flat<int8>().data();

    // in label
    const Tensor& in_label = context->input(1);
    auto in_label_ptr = in_label.flat<int>().data();

    // out octree
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, in_octree.shape(), &out_tensor));
    auto out_ptr = out_tensor->flat<int8>().data();
    cudaMemcpy(out_ptr, in_ptr, in_octree.NumElements(), cudaMemcpyDeviceToDevice);

    // parse octree info
    OctreeInfo oct_info;
    cudaMemcpy(&oct_info, out_ptr, sizeof(OctreeInfo), cudaMemcpyDeviceToHost);
    OctreeParser octree_;
    octree_.set_gpu(out_ptr, &oct_info);
    int node_num = octree_.info().node_num(curr_depth_);
    CHECK_EQ(node_num, in_label.NumElements());

    // update children
    int split_num = 0;  // non-empty node number
    int* children = octree_.mutable_children_gpu(curr_depth_);
    generate_label_gpu(children, split_num, in_label_ptr, node_num, mask_);

    // deal with degenatated case
    if (split_num == 0) {
      split_num = 1;
      memset_gpu(1, 0, children);
      LOG(INFO) << "Warning: split_num == 0 in octree update layer.";
    }

    oct_info.set_nempty(curr_depth_, split_num);
    cudaMemcpy(out_ptr, &oct_info, sizeof(OctreeInfo), cudaMemcpyHostToDevice);
  }

 private:
  int curr_depth_;
  int mask_;
};

REGISTER_KERNEL_BUILDER(Name("OctreeUpdate").Device(DEVICE_GPU), OctreeUpdateOp);

}  // namespace tensorflow
