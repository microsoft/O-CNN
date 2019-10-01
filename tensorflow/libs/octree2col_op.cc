#include "octree_nn.h"
#include "octree_parser.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("OctreeToCol")
    .Input("btm_data: float")
    .Input("in_octree: int8")
    .Attr("depth: int")
    .Attr("kernel_size: list(int)")
    .Attr("stride: int")
    .Output("top_data: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // (1, C, H, 1) -> (C, kernel_dim, H')
      vector<int> kernel_size;
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));
      resize_with_last_val(kernel_size, 3);
      int kernel_dim = num_elements(kernel_size);
      c->set_output(0,
          c->MakeShape({c->Dim(c->input(0), 1), kernel_dim, c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(Octree2col operator.)doc");


REGISTER_OP("ColToOctree")
    .Input("top_grad: float")
    .Input("in_octree: int8")
    .Attr("depth: int")
    .Attr("kernel_size: list(int)")
    .Attr("stride: int")
    .Output("btm_grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // (C, kernel_dim, H) -> (1, C, H', 1)
      c->set_output(0,
          c->MakeShape({1, c->Dim(c->input(0), 0), c->UnknownDim(), 1}));
      return Status::OK();
    })
    .Doc(R"doc(Gradient for octree convolution operator.)doc");


class Octree2ColBase : public OpKernel {
 public:
  explicit Octree2ColBase(OpKernelConstruction* context)
    : OpKernel(context) {
    // get attributes
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size_));
    resize_with_last_val(kernel_size_, 3);

    CHECK_GT(depth_, 0) << "Depth should be larger than 0";
    for(auto k : kernel_size_) { CHECK(0 < k && k < 4) << "Invalide kernel size"; }
    CHECK(stride_ == 1 || stride_ == 2) << "Unsupport stride";
  }

  void init_ni_ptr(OpKernelContext* ctx, Tensor& ni_gpu) {
    vector<int>& ni_cpu = NeighHelper::Get().get_ni(kernel_size_);
    int count = ni_cpu.size();
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({ count }), &ni_gpu));
    // int* ni_ptr = ni_gpu.flat<int>().data();
    // cudaMemcpy(ni_ptr, ni_cpu.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
    memcpy_gpu(count, ni_cpu.data(), ni_gpu.flat<int>().data());
  }

  void set_octree_parser(OpKernelContext* context, OctreeParser& octree_) {
    auto octree_ptr = context->input(1).flat<int8>().data();
    octree_.set_gpu(octree_ptr);
  }

 protected:
  int depth_;
  int stride_;
  vector<int> kernel_size_;
};


class OctreeToColOp : public Octree2ColBase {
 public:
  explicit OctreeToColOp(OpKernelConstruction* context)
    : Octree2ColBase(context) {}

  void Compute(OpKernelContext* context) override {
    // init
    OctreeParser octree_;
    this->set_octree_parser(context, octree_);
    Tensor ni_gpu;
    this->init_ni_ptr(context, ni_gpu);
    auto ni_ptr =ni_gpu.flat<int>().data();

    // input data, data format: [1, channels, H, 1]
    const Tensor& btm_data = context->input(0);
    const TensorShape& btm_shape = btm_data.shape();
    int btm_depth = this->depth_;
    int channel = btm_shape.dim_size(1);
    int btm_height = btm_shape.dim_size(2);
    CHECK_EQ(octree_.info().node_num(btm_depth), btm_height);

    // output data
    int top_height = btm_height;
    if (this->stride_ == 2) {
      top_height = btm_height / 8;
      int top_depth = btm_depth - 1;
      CHECK_EQ(top_height, octree_.info().node_num_nempty(top_depth));
    }
    Tensor* top_data = nullptr;
    int kernel_sdim = num_elements(this->kernel_size_);
    TensorShape top_shape({channel, kernel_sdim, top_height});
    OP_REQUIRES_OK(context, context->allocate_output(0, top_shape, &top_data));

    // execute
    auto btm_ptr = btm_data.flat<float>().data();
    auto top_ptr = top_data->flat<float>().data();
    octree2col_gpu(top_ptr, btm_ptr, channel, top_height,
        kernel_sdim, this->stride_, octree_.neighbor_gpu(btm_depth),
        ni_ptr, top_height, 0);
  }
};


class ColToOctreeOp : public Octree2ColBase {
 public:
  explicit ColToOctreeOp(OpKernelConstruction* context)
    : Octree2ColBase(context) {}

  void Compute(OpKernelContext* context) override {
    // init
    OctreeParser octree_;
    this->set_octree_parser(context, octree_);
    Tensor ni_gpu;
    this->init_ni_ptr(context, ni_gpu);
    auto ni_ptr =ni_gpu.flat<int>().data();

    // in grad
    const Tensor& top_grad = context->input(0);
    const TensorShape& top_shape = top_grad.shape();
    int channel = top_shape.dim_size(0);
    int top_height = top_shape.dim_size(2);

    // out grad
    int btm_depth = this->depth_;
    int btm_height = octree_.info().node_num(btm_depth);
    if (this->stride_ == 2) {
      CHECK_EQ(top_height, octree_.info().node_num_nempty(btm_depth - 1));
    }
    Tensor* btm_grad = nullptr;
    TensorShape btm_shape({1, channel, btm_height, 1});
    OP_REQUIRES_OK(context, context->allocate_output(0, btm_shape, &btm_grad));

    // execute
    auto top_ptr = top_grad.flat<float>().data();
    auto btm_ptr = btm_grad->flat<float>().data();
    // int kernel_size = this->kernel_size_;
    int kernel_sdim = num_elements(this->kernel_size_);
    col2octree_gpu(top_ptr, btm_ptr, channel, top_height,
        kernel_sdim, this->stride_, octree_.neighbor_gpu(btm_depth),
        ni_ptr, top_height, 0);
  }
};


REGISTER_KERNEL_BUILDER(Name("OctreeToCol").Device(DEVICE_GPU), OctreeToColOp);
REGISTER_KERNEL_BUILDER(Name("ColToOctree").Device(DEVICE_GPU), ColToOctreeOp);

}  // namespace tensorflow
