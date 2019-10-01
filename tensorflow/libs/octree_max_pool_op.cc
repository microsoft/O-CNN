#include "octree_nn.h"
#include "octree_parser.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>


namespace tensorflow {

auto pool_shape_fun = [](::tensorflow::shape_inference::InferenceContext* c) {
  auto top_shape = c->input(0);
  TF_RETURN_IF_ERROR(c->ReplaceDim(top_shape, 2, c->UnknownDim(), &top_shape));
  c->set_output(0, top_shape);
  return Status::OK();
};

auto pool_shape_fun2 = [](::tensorflow::shape_inference::InferenceContext* c) {
  auto top_shape = c->input(0);
  TF_RETURN_IF_ERROR(c->ReplaceDim(top_shape, 2, c->UnknownDim(), &top_shape));
  c->set_output(0, top_shape);
  c->set_output(1, top_shape);
  return Status::OK();
};


REGISTER_OP("OctreeMaxPool")
    .Input("btm_data: float")
    .Input("octree: int8")
    .Attr("depth: int")
    .Output("top_data: float")
    .Output("mask: int32")
    .SetShapeFn(pool_shape_fun2)
    .Doc(R"doc(Octree max pooling operator.)doc");

REGISTER_OP("OctreeMaxUnpool")
    .Input("top_data: float")
    .Input("mask: int32")
    .Input("octree: int8")
    .Attr("depth: int")
    .Output("btm_data: float")
    .SetShapeFn(pool_shape_fun)
    .Doc(R"doc(Octree max unpooling operator.)doc");

REGISTER_OP("OctreeMaskPool")
    .Input("btm_data: float")
    .Input("mask: int32")
    .Input("octree: int8")
    .Attr("depth: int")
    .Output("top_data: float")
    .SetShapeFn(pool_shape_fun)
    .Doc(R"doc(Octree mask pooling operator.)doc");


class OctreePoolBase : public OpKernel {
 public:
  explicit OctreePoolBase(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    CHECK_GT(depth_, 1) << "Depth should be larger than 1";
  }

  void set_octree_parser(OpKernelContext* context, int idx, OctreeParser& octree_) {
    auto octree_ptr = context->input(idx).flat<int8>().data();
    octree_.set_gpu(octree_ptr);
  }

 protected:
  int depth_;
};

class OctreeMaxPoolOp : public OctreePoolBase {
 public:
  explicit OctreeMaxPoolOp(OpKernelConstruction* context)
    : OctreePoolBase(context) {}

  void Compute(OpKernelContext* context) override {
    // in octree
    OctreeParser octree_;
    this->set_octree_parser(context, 1, octree_);

    // btm data
    const Tensor& btm_data = context->input(0);
    const TensorShape& btm_shape = btm_data.shape();
    auto btm_ptr = btm_data.flat<float>().data();
    int channel = btm_shape.dim_size(1);
    int btm_h = btm_shape.dim_size(2);

    // check
    int btm_depth = this->depth_;
    CHECK_EQ(octree_.info().node_num(btm_depth), btm_h);

    // top data
    TensorShape top_shape = btm_shape;
    int top_h = btm_h >> 3;
    top_shape.set_dim(2, top_h);
    Tensor* top_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, top_shape, &top_data));
    auto top_ptr = top_data->flat<float>().data();

    // mask
    Tensor* mask = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, top_shape, &mask));
    auto mask_ptr = mask->flat<int>().data();

    // pooling
    octree_max_pool_gpu(top_ptr, top_h, mask_ptr, btm_ptr, btm_h, channel);
  }
};

class OctreeMaxUnpoolOp : public OctreePoolBase {
 public:
  explicit OctreeMaxUnpoolOp(OpKernelConstruction* context)
    : OctreePoolBase(context) {}

  void Compute(OpKernelContext* context) override {
    // in octree
    OctreeParser octree_;
    this->set_octree_parser(context, 2, octree_);

    // top data
    const Tensor& top_data = context->input(0);
    const TensorShape& top_shape = top_data.shape();
    auto top_ptr = top_data.flat<float>().data();
    int channel = top_shape.dim_size(1);
    int top_h = top_shape.dim_size(2);

    // mask
    const Tensor& mask = context->input(1);
    const TensorShape& mask_shape = mask.shape();
    auto mask_ptr = mask.flat<int>().data();

    // check
    int btm_depth = this->depth_;
    CHECK(mask_shape == top_shape);
    CHECK_EQ(top_h, octree_.info().node_num_nempty(btm_depth - 1));

    // top data
    TensorShape btm_shape = top_shape;
    int btm_h = top_h << 3;
    btm_shape.set_dim(2, btm_h);
    Tensor* btm_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, btm_shape, &btm_data));
    auto btm_ptr = btm_data->flat<float>().data();

    // pooling
    octree_max_unpool_gpu(top_ptr, top_h, mask_ptr, btm_ptr, btm_h, channel);
  }
};

class OctreeMaskPoolOp : public OctreePoolBase {
 public:
  explicit OctreeMaskPoolOp(OpKernelConstruction* context)
    : OctreePoolBase(context) {}

  void Compute(OpKernelContext* context) override {
    // in octree
    OctreeParser octree_;
    this->set_octree_parser(context, 2, octree_);

    // btm data
    const Tensor& btm_data = context->input(0);
    const TensorShape& btm_shape = btm_data.shape();
    auto btm_ptr = btm_data.flat<float>().data();
    int channel = btm_shape.dim_size(1);
    int btm_h = btm_shape.dim_size(2);

    // mask
    const Tensor& mask = context->input(1);
    const TensorShape& top_shape = mask.shape();
    auto mask_ptr = mask.flat<int>().data();
    int top_h = top_shape.dim_size(2);

    // check
    int btm_depth = this->depth_;
    CHECK_EQ(octree_.info().node_num(btm_depth), btm_h);
    CHECK_EQ(top_h, btm_h >> 3);

    // top data
    Tensor* top_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, top_shape, &top_data));
    auto top_ptr = top_data->flat<float>().data();

    // pooling
    octree_mask_pool_gpu(top_ptr, top_h, mask_ptr, btm_ptr, btm_h, channel);
  }
};


REGISTER_KERNEL_BUILDER(Name("OctreeMaxPool").Device(DEVICE_GPU), OctreeMaxPoolOp);
REGISTER_KERNEL_BUILDER(Name("OctreeMaxUnpool").Device(DEVICE_GPU), OctreeMaxUnpoolOp);
REGISTER_KERNEL_BUILDER(Name("OctreeMaskPool").Device(DEVICE_GPU), OctreeMaskPoolOp);

}  // namespace tensorflow
