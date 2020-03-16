#include "octree_parser.h"
#include "octree_nn.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

auto pad_shape_fun = [](::tensorflow::shape_inference::InferenceContext* c) {
  auto top_shape = c->input(0);
  TF_RETURN_IF_ERROR(c->ReplaceDim(top_shape, 2, c->UnknownDim(), &top_shape));
  c->set_output(0, top_shape);
  return Status::OK();
};

REGISTER_OP("OctreePad")
    .Input("btm_data: float")
    .Input("octree: int8")
    .Attr("depth: int")
    .Attr("dval: float = 0.0")
    .Output("top_data: float")
    .SetShapeFn(pad_shape_fun)
    .Doc(R"doc(Octree padding operator.)doc");

REGISTER_OP("OctreeDepad")
    .Input("top_data: float")
    .Input("octree: int8")
    .Attr("depth: int")
    .Output("btm_data: float")
    .SetShapeFn(pad_shape_fun)
    .Doc(R"doc(Octree depadding operator.)doc");


class OctreePadBase : public OpKernel {
 public:
  explicit OctreePadBase(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    CHECK_GE(depth_, 1) << "Depth should be larger than 1";
  }

  void set_octree_parser(OpKernelContext* context, OctreeParser& octree_) {
    auto octree_ptr = context->input(1).flat<int8>().data();
    octree_.set_gpu(octree_ptr);
  }

 protected:
  int depth_;
};


class OctreePadOp : public OctreePadBase {
 public:
  explicit OctreePadOp(OpKernelConstruction* context)
    : OctreePadBase(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dval", &dval_));
  }

  void Compute(OpKernelContext* context) override {
    // in octree
    OctreeParser octree_;
    this->set_octree_parser(context, octree_);

    // btm data
    const Tensor& btm_data = context->input(0);
    const TensorShape& btm_shape = btm_data.shape();
    auto btm_ptr = btm_data.flat<float>().data();
    int channel = btm_shape.dim_size(1);
    int btm_h = btm_shape.dim_size(2);

    // check
    int depth = this->depth_;
    CHECK_EQ(octree_.info().node_num_nempty(depth), btm_h)
        << ", pad, d = " << depth << ", channel = " << channel;

    // top data
    TensorShape top_shape = btm_shape;
    int top_h = octree_.info().node_num(depth);
    top_shape.set_dim(2, top_h);
    Tensor* top_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, top_shape, &top_data));
    auto top_ptr = top_data->flat<float>().data();

    // padding data
    pad_forward_gpu(top_ptr, top_h, channel,
        btm_ptr, btm_h, octree_.children_gpu(depth), dval_);
  }
  
 protected:
  float dval_;
};


class OctreeDepadOp : public OctreePadBase {
 public:
  explicit OctreeDepadOp(OpKernelConstruction* context)
    : OctreePadBase(context) {}

  void Compute(OpKernelContext* context) override {
    // in octree
    OctreeParser octree_;
    this->set_octree_parser(context, octree_);

    // top grad
    const Tensor& top_data = context->input(0);
    const TensorShape& top_shape = top_data.shape();
    auto top_ptr = top_data.flat<float>().data();
    int channel = top_shape.dim_size(1);
    int top_h = top_shape.dim_size(2);

    // check
    int depth = this->depth_;
    CHECK_EQ(octree_.info().node_num(depth), top_h)
        << ", depad, d = " << depth << ", channel = " << channel;

    // btm grad
    TensorShape btm_shape = top_shape;
    int btm_h = octree_.info().node_num_nempty(depth);
    btm_shape.set_dim(2, btm_h);
    Tensor* btm_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, btm_shape, &btm_data));
    auto btm_ptr = btm_data->flat<float>().data();

    // padding data
    pad_backward_gpu(btm_ptr, btm_h, channel,
        top_ptr, top_h, octree_.children_gpu(depth));
  }
};


REGISTER_KERNEL_BUILDER(Name("OctreePad").Device(DEVICE_GPU), OctreePadOp);
REGISTER_KERNEL_BUILDER(Name("OctreeDepad").Device(DEVICE_GPU), OctreeDepadOp);

}  // namespace tensorflow
