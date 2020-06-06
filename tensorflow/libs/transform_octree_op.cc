#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "transform_octree.h"

namespace tensorflow {

REGISTER_OP("OctreeDrop")
    .Input("octree_in: string")
    .Input("depth: int32")
    .Input("ratio: float")
    .Output("octree_out: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(Drop out octree nodes.)doc");

REGISTER_OP("OctreeScan")
    .Input("octree_in: string")
    .Input("axis: float")
    .Attr("scale: float=1.0")
    .Output("octree_out: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(Drop octree nodes via scanning.)doc");

REGISTER_OP("OctreeCast")
    .Input("octree_in: int8")
    .Output("octree_out: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({1}));
      return Status::OK();
    })
    .Doc(R"doc(Cast the octree tensor from `int8` to `string`.)doc");

class OctreeDropOp : public OpKernel {
 public:
  explicit OctreeDropOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // input
    const Tensor& data_in = context->input(0);
    const string& octree_in = data_in.flat<string>()(0);
    int depth = context->input(1).flat<int>()(0);
    float ratio = context->input(2).flat<float>()(0);

    vector<char> octree_out;
    octree_dropout(octree_out, octree_in, depth, ratio);

    // output
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, data_in.shape(), &data_out));
    string& str_out = data_out->flat<string>()(0);
    str_out.assign(octree_out.begin(), octree_out.end());
  }
};

class OctreeScanOp : public OpKernel {
 public:
  explicit OctreeScanOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

  void Compute(OpKernelContext* context) override {
    // input
    OctreeParser octree_in;
    const Tensor& data_in = context->input(0);
    octree_in.set_cpu(data_in.flat<string>()(0).data());

    const Tensor& axis_in = context->input(1);
    auto ptr_in = axis_in.flat<float>().data();
    vector<float> axis(ptr_in, ptr_in + axis_in.NumElements());

    ScanOctree scan_octree(scale_);
    vector<char> octree_out;
    scan_octree.scan(octree_out, octree_in, axis);

    // output
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, data_in.shape(), &data_out));
    string& str_out = data_out->flat<string>()(0);
    str_out.assign(octree_out.begin(), octree_out.end());
  }

 protected:
  float scale_;
};

class OctreeCastOp : public OpKernel {
 public:
  explicit OctreeCastOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // input
    const Tensor& data_in = context->input(0);
    const char* ptr_in = (const char*)data_in.flat<int8>().data();

    // output
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &data_out));
    string& str_out = data_out->flat<string>()(0);
    str_out.assign(ptr_in, ptr_in + data_in.NumElements());
  }
};

REGISTER_KERNEL_BUILDER(Name("OctreeCast").Device(DEVICE_CPU), OctreeCastOp);
REGISTER_KERNEL_BUILDER(Name("OctreeScan").Device(DEVICE_CPU), OctreeScanOp);
REGISTER_KERNEL_BUILDER(Name("OctreeDrop").Device(DEVICE_CPU), OctreeDropOp);

}  // namespace tensorflow
