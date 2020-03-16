// #include "octree.h"
#include "transform_octree.h"
#include "math_functions.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>


namespace tensorflow {

REGISTER_OP("OctreeDropout")
    .Input("octree_in: string")
    .Input("depth: int32")
    .Input("ratio: float")
    .Output("octree_out: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({ c->UnknownDim() }));
      return Status::OK();
    })
    .Doc(R"doc(Drop out octree nodes.)doc");


class OctreeDropoutOp : public OpKernel {
 public:
  explicit OctreeDropoutOp(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // // input
    // const Tensor& data_in = context->input(0);
    // // Octree octree_in;
    // // octree_in.set_cpu(data_in.flat<string>()(0).data());
    // const string& octree_in = data_in.flat<string>()(0);
    // int depth = context->input(1).flat<int>()(0);
    // float ratio = context->input(2).flat<float>()(0);

    // // check
    // string msg;
    // bool succ = octree_in.info().check_format(msg);
    // CHECK(succ) << msg;
    // CHECK_EQ(data_in.NumElements(), 1);
    // CHECK(0.0 <= ratio && ratio < 1.0) << "Invalid ratio: " << ratio;
    // CHECK(1 < depth && depth < octree_in.info().depth()) << "Invalid depth: " << depth;

    // // dropout
    // Octree octree_out;
    // octree_in.dropout(octree_out, depth, ratio);

    // // output
    // Tensor* data_out = nullptr;
    // OP_REQUIRES_OK(context, context->allocate_output(0, data_in.shape(), &data_out));
    // string& str_out = data_out->flat<string>()(0);
    // // const char* ptr = octree_out.ptr_raw_cpu();
    // // str_out.assign(ptr, ptr + octree_out.info().sizeof_octree());
    // // const char* ptr = octree_in.ptr_raw_cpu();
    // // str_out.assign(ptr, ptr + octree_in.info().sizeof_octree());

    // input
    const Tensor& data_in = context->input(0);
    const string& octree_in = data_in.flat<string>()(0);
    int depth = context->input(1).flat<int>()(0);
    float ratio = context->input(2).flat<float>()(0);

    vector<char> octree_out;
    octree_dropout(octree_out, octree_in, depth, ratio); 

    // output
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, data_in.shape(), &data_out));
    string& str_out = data_out->flat<string>()(0);
    str_out.assign(octree_out.begin(), octree_out.end());  
    // str_out = octree_in;    
  }
};

REGISTER_KERNEL_BUILDER(Name("OctreeDropout").Device(DEVICE_CPU), OctreeDropoutOp);

}  // namespace tensorflow
