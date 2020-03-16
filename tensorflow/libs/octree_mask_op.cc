#include "octree_nn.h"

#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("OctreeMask")
    .Input("in_data: float")
    .Input("in_label: int32")
    .Attr("mask: int")
    .Output("out_data: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Octree mask operator.)doc");


class OctreeMaskOp : public OpKernel {
 public:
  explicit OctreeMaskOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("mask", &mask_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& in_data = context->input(0);
    const TensorShape in_shape = in_data.shape();
    auto in_ptr = in_data.flat<float>().data();

    // in label
    const Tensor& in_label = context->input(1);
    auto label_ptr = in_label.flat<int>().data();
    CHECK_EQ(in_shape.dim_size(2), in_label.NumElements());

    // out data
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, in_shape, &out_tensor));
    auto out_ptr = out_tensor->flat<float>().data();

    // exec
    int height = in_shape.dim_size(2);
    int channel = in_shape.dim_size(1);
    int num = channel * height;
    octree_mask_gpu(out_ptr, in_ptr, label_ptr, height, mask_, num);
  }

 private:
  int mask_;
};


REGISTER_KERNEL_BUILDER(Name("OctreeMask").Device(DEVICE_GPU), OctreeMaskOp);

}  // namespace tensorflow
