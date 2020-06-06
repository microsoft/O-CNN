#include "octree_nn.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

auto gather_shape_fun = [](::tensorflow::shape_inference::InferenceContext* c) {
  auto top_shape = c->input(0);
  TF_RETURN_IF_ERROR(c->ReplaceDim(top_shape, 2, c->UnknownDim(), &top_shape));
  c->set_output(0, top_shape);
  return Status::OK();
};

REGISTER_OP("OctreeGather")
    .Input("btm_data: float")   // (1, C, Hb, 1)
    .Input("index: int32")      // (Ht,)
    .Output("top_data: float")  // (1, C, Ht, 1)
    .SetShapeFn(gather_shape_fun)
    .Doc(R"doc(Octree gather operator.)doc");

REGISTER_OP("OctreeGatherbk")
    .Input("top_data: float")  // (1, C, Ht, 1)
    .Input("index: int32")     // (Ht,)
    .Input("btm_shape: int32") // (4,)
    .Output("btm_data: float") // (1, C, Hb, 1)
    .SetShapeFn(gather_shape_fun)
    .Doc(R"doc(Octree gather backward operator.)doc");


class OctreeGatherOp : public OpKernel {
 public:
  explicit OctreeGatherOp(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // btm data
    const Tensor& btm_tensor = context->input(0);
    const TensorShape& btm_shape = btm_tensor.shape();
    auto btm_data = btm_tensor.flat<float>().data();
    int channel = btm_shape.dim_size(1);
    int btm_h = btm_shape.dim_size(2);

    // index data
    const Tensor& idx_tensor = context->input(1);
    auto idx = idx_tensor.flat<int>().data();
    int top_h = idx_tensor.dim_size(0);

    // top data
    TensorShape top_shape = btm_shape;
    top_shape.set_dim(2, top_h);
    Tensor* top_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, top_shape, &top_tensor));
    auto top_data = top_tensor->flat<float>().data();

    // gather data
    octree_gather_gpu(top_data, top_h, channel, btm_data, btm_h, idx);
  }
};


class OctreeGatherbkOp : public OpKernel {
 public:
  explicit OctreeGatherbkOp(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // top grad
    const Tensor& top_tensor = context->input(0);
    const TensorShape& top_shape = top_tensor.shape();
    auto top_data = top_tensor.flat<float>().data();
    int channel = top_shape.dim_size(1);
    int top_h = top_shape.dim_size(2);

    // index data
    const Tensor& idx_tensor = context->input(1);
    auto idx = idx_tensor.flat<int>().data();
    CHECK_EQ(top_h, idx_tensor.dim_size(0));

    // shape
    const Tensor& shape_tensor = context->input(2);
    auto shape_data = shape_tensor.flat<int>().data();
    int btm_h = shape_data[2];
    CHECK(shape_tensor.NumElements() == 4 && shape_data[0] == 1 &&
          shape_data[1] == channel && shape_data[3] == 1);

    // btm grad
    TensorShape btm_shape = top_shape;
    btm_shape.set_dim(2, btm_h);
    Tensor* btm_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, btm_shape, &btm_tensor));
    auto btm_data = btm_tensor->flat<float>().data();

    // padding data
    octree_gatherbk_gpu(top_data, top_h, channel, btm_data, btm_h, idx);
  }
};


REGISTER_KERNEL_BUILDER(Name("OctreeGather").Device(DEVICE_GPU), OctreeGatherOp);
REGISTER_KERNEL_BUILDER(Name("OctreeGatherbk").Device(DEVICE_GPU).HostMemory("btm_shape"), OctreeGatherbkOp);

}  // namespace tensorflow
