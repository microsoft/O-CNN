#include "octree_conv.h"
#include "octree_nn.h"
#include "tensorflow_gpu_gemm.h"

#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {
using octree::OctreeBaseConv;

auto conv_forward_fun = [](::tensorflow::shape_inference::InferenceContext* c) {
  int num_output;
  TF_RETURN_IF_ERROR(c->GetAttr("num_output", &num_output));
  c->set_output(0, c->MakeShape({ 1, num_output, c->UnknownDim(), 1 }));
  return Status::OK();
};

auto conv_backward_fun = [](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  c->set_output(1, c->input(1));
  return Status::OK();
};


REGISTER_OP("OctreeConv")
    .Input("btm_data: float")
    .Input("weights: float")
    .Input("octree: int8")
    .Attr("depth: int")
    .Attr("num_output: int")
    .Attr("kernel_size: list(int)")
    .Attr("stride: int")
    .Output("top_data: float")
    .SetShapeFn(conv_forward_fun)
    .Doc(R"doc(Octree convolution operator.)doc");

REGISTER_OP("OctreeDeconv")
    .Input("btm_data: float")
    .Input("weights: float")
    .Input("octree: int8")
    .Attr("depth: int")
    .Attr("num_output: int")
    .Attr("kernel_size: list(int)")
    .Attr("stride: int")
    .Output("top_data: float")
    .SetShapeFn(conv_forward_fun)
    .Doc(R"doc(Octree deconvolution operator.)doc");

REGISTER_OP("OctreeConvGrad")
    .Input("btm_data: float")
    .Input("weights: float")
    .Input("octree: int8")
    .Input("top_diff: float")
    .Attr("depth: int")
    .Attr("num_output: int")
    .Attr("kernel_size: list(int)")
    .Attr("stride: int")
    .Output("btm_diff: float")
    .Output("weight_diff: float")
    .SetShapeFn(conv_backward_fun)
    .Doc(R"doc(Gradient for octree convolution operator.)doc");

REGISTER_OP("OctreeDeconvGrad")
    .Input("btm_data: float")
    .Input("weights: float")
    .Input("octree: int8")
    .Input("top_diff: float")
    .Attr("depth: int")
    .Attr("num_output: int")
    .Attr("kernel_size: list(int)")
    .Attr("stride: int")
    .Output("btm_diff: float")
    .Output("weight_diff: float")
    .SetShapeFn(conv_backward_fun)
    .Doc(R"doc(Gradient for octree convolution operator.)doc");


class OctreeConvTF : public OpKernel, public OctreeBaseConv<float> {
 public:
  explicit OctreeConvTF(OpKernelConstruction* context)
    : OpKernel(context), OctreeBaseConv<float>() {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("num_output", &num_output_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size_));
    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
    resize_with_last_val(kernel_size_, 3);

    CHECK_GT(depth_, 0) << "The depth should be larger than 0";
    CHECK_GT(num_output_, 0) << "The num_output should be larger than 0";
    for (auto k : kernel_size_) { CHECK(0 < k && k < 4) << "Invalide kernel size"; }
    CHECK(stride_ == 1 || stride_ == 2) << "Unsupport stride";
  }

  void setup_op(OpKernelContext* context) {
    // setup gemm
    tf_gemm_gpu_.set_context(context);
    this->engine_gpu_ = &tf_gemm_gpu_;

    // setup octree
    auto in_octree_ptr = context->input(2).flat<int8>().data();
    this->octree_.set_gpu(in_octree_ptr);

    // setup octree conv
    const TensorShape& shape_in = context->input(0).shape();
    int channel_in = shape_in.dim_size(1), height_btm = shape_in.dim_size(2);
    OctreeBaseConv<float>::setup(kernel_size_, stride_, depth_,
        channel_in, num_output_, false);
    if (stride_ == 2 && is_deconvolution_layer()) {
      CHECK_EQ(height_btm, this->octree_.info().node_num_nempty(depth_));
    } else {
      CHECK_EQ(height_btm, this->octree_.info().node_num(depth_))
          << ", d: " << depth_ << ", channel_in: " << channel_in;
    }
  }

  void alloc_temp_memory(OpKernelContext* ctx, Tensor* workspace,
      Tensor* data_buffer, Tensor* result_buffer, Tensor* ni_gpu) {
    OctreeBaseConv<float>::reshape();

    int count = num_elements(this->workspace_shape_);
    OP_REQUIRES_OK(ctx,
        ctx->allocate_temp(DT_FLOAT, TensorShape({ count }), workspace));
    this->workspace_ = workspace->flat<float>().data();

    count = num_elements(this->result_buffer_shape_);
    if (count != 0) {
      OP_REQUIRES_OK(ctx,
          ctx->allocate_temp(DT_FLOAT, TensorShape({ count }), result_buffer));
      this->result_buffer_ = result_buffer->flat<float>().data();
    } else {
      this->result_buffer_ = nullptr;
    }

    // count = num_elements(this->data_buffer_shape_);
    // if (count != 0) {
    //   OP_REQUIRES_OK(ctx,
    //       ctx->allocate_temp(DT_FLOAT, TensorShape({ count }), data_buffer));
    //   this->data_buffer_ = data_buffer->flat<float>().data();
    // } else {
    //   this->data_buffer_ = nullptr;
    // }

    vector<int>& ni_cpu = NeighHelper::get_ni(kernel_size_);
    count = ni_cpu.size();
    if (count != 0) {
      OP_REQUIRES_OK(ctx,
          ctx->allocate_temp(DT_INT32, TensorShape({ count }), ni_gpu));
      auto ni_ptr = ni_gpu->flat<int>().data();
      cudaMemcpy(ni_ptr, ni_cpu.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
      this->ni_gpu_ptr_ = ni_ptr;
    }
  }

 private:
  int depth_;
  int num_output_;
  int stride_;
  vector<int> kernel_size_;
  GEMMEngineTF tf_gemm_gpu_;
};


class OctreeConvOp : public OctreeConvTF {
 public:
  explicit OctreeConvOp(OpKernelConstruction* context)
    : OctreeConvTF(context) {}

  void Compute(OpKernelContext* context) override {
    // init
    this->setup_op(context);
    Tensor workshape, data_buffer, rst_buffer, ni_gpu, *data_out;
    this->alloc_temp_memory(context, &workshape, &data_buffer, &rst_buffer, &ni_gpu);
    alloc_output_memory(context, &data_out);

    // get points
    auto btm_data = context->input(0).flat<float>().data();
    auto weights = context->input(1).flat<float>().data();
    auto top_data = data_out->flat<float>().data();

    // forward
    this->forward_gpu_gemm(top_data, btm_data, weights);
  }

  virtual bool is_deconvolution_layer() override { return false; }

  void alloc_output_memory(OpKernelContext* context, Tensor** data_out) {
    TensorShape tshape({ 1, this->top_shape_[1], this->top_shape_[2], 1 });
    OP_REQUIRES_OK(context, context->allocate_output(0, tshape, data_out));
  }
};


class OctreeConvGradOp : public OctreeConvTF {
 public:
  explicit OctreeConvGradOp(OpKernelConstruction* context)
    : OctreeConvTF(context) {}

  void Compute(OpKernelContext* context) override {
    // init
    this->setup_op(context);
    Tensor workshape, data_buffer, rst_buffer, ni_gpu, *btm_out, *weights_out;
    this->alloc_temp_memory(context, &workshape, &data_buffer, &rst_buffer, &ni_gpu);
    alloc_output_memory(context, &btm_out, &weights_out);

    // get points
    auto btm_data = context->input(0).flat<float>().data();
    auto weights = context->input(1).flat<float>().data();
    auto top_diff = context->input(3).flat<float>().data();
    auto btm_diff = btm_out->flat<float>().data();
    auto weights_diff = weights_out->flat<float>().data();

    // backward
    this->weight_gpu_gemm(weights_diff, btm_data, top_diff);
    this->backward_gpu_gemm(btm_diff, top_diff, weights);
  }

  virtual bool is_deconvolution_layer() { return false; }

  void alloc_output_memory(OpKernelContext* context,
      Tensor** btm_out, Tensor** weights_out) {
    OP_REQUIRES_OK(context,
        context->allocate_output(0, context->input(0).shape(), btm_out));
    OP_REQUIRES_OK(context,
        context->allocate_output(1, context->input(1).shape(), weights_out));
  }
};


class OctreeDeconvOp : public OctreeConvTF {
 public:
  explicit OctreeDeconvOp(OpKernelConstruction* context)
    : OctreeConvTF(context) {}

  void Compute(OpKernelContext* context) override {
    // init
    this->setup_op(context);
    Tensor workshape, data_buffer, rst_buffer, ni_gpu, *data_out;
    this->alloc_temp_memory(context, &workshape, &data_buffer, &rst_buffer, &ni_gpu);
    alloc_output_memory(context, &data_out);

    // get points
    auto btm_data = context->input(0).flat<float>().data();
    auto weights = context->input(1).flat<float>().data();
    auto top_data = data_out->flat<float>().data();

    // forward
    this->backward_gpu_gemm(top_data, btm_data, weights);
  }

  virtual bool is_deconvolution_layer() override { return true; }

  void alloc_output_memory(OpKernelContext* context, Tensor** data_out) {
    TensorShape tshape({ 1, this->top_shape_[1], this->top_shape_[2], 1 });
    OP_REQUIRES_OK(context, context->allocate_output(0, tshape, data_out));
  }
};


class OctreeDeconvGradOp : public OctreeConvTF {
 public:
  explicit OctreeDeconvGradOp(OpKernelConstruction* context)
    : OctreeConvTF(context) {}

  void Compute(OpKernelContext* context) override {
    // init
    this->setup_op(context);
    Tensor workshape, data_buffer, rst_buffer, ni_gpu, *btm_out, *weights_out;
    this->alloc_temp_memory(context, &workshape, &data_buffer, &rst_buffer, &ni_gpu);
    alloc_output_memory(context, &btm_out, &weights_out);

    // get points
    auto btm_data = context->input(0).flat<float>().data();
    auto weights = context->input(1).flat<float>().data();
    auto top_diff = context->input(3).flat<float>().data();
    auto btm_diff = btm_out->flat<float>().data();
    auto weights_diff = weights_out->flat<float>().data();

    // backward
    this->weight_gpu_gemm(weights_diff, top_diff, btm_data);
    this->forward_gpu_gemm(btm_diff, top_diff, weights);
  }

  virtual bool is_deconvolution_layer() { return true; }

  void alloc_output_memory(OpKernelContext* context,
      Tensor** btm_out, Tensor** weights_out) {
    OP_REQUIRES_OK(context,
        context->allocate_output(0, context->input(0).shape(), btm_out));
    OP_REQUIRES_OK(context,
        context->allocate_output(1, context->input(1).shape(), weights_out));
  }
};


REGISTER_KERNEL_BUILDER(Name("OctreeConv").Device(DEVICE_GPU), OctreeConvOp);
REGISTER_KERNEL_BUILDER(Name("OctreeDeconv").Device(DEVICE_GPU), OctreeDeconvOp);
REGISTER_KERNEL_BUILDER(Name("OctreeConvGrad").Device(DEVICE_GPU), OctreeConvGradOp);
REGISTER_KERNEL_BUILDER(Name("OctreeDeconvGrad").Device(DEVICE_GPU), OctreeDeconvGradOp);

}  // namespace tensorflow
