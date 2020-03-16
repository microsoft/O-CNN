#include "octree_samples.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>


namespace tensorflow {

REGISTER_OP("OctreeSamples")
    .Input("names: string")
    .Output("octrees: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Get one sample octree for testing.)doc");


class OctreeSamplesOp : public OpKernel {
 public:
  explicit OctreeSamplesOp(OpKernelConstruction* context) :
    OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& names = context->input(0);
    int num = names.NumElements();
    CHECK_GE(num, 1);

    Tensor* octrees = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, names.shape(), &octrees));

    for (int i = 0; i < num; ++i) {
      string name = names.flat<string>()(i);
      string& oct = octrees->flat<string>()(i);
      
      size_t size = 0;
      const char* str = (const char*)octree::get_one_octree(name.c_str(), &size);
      oct.assign(str, str + size);
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("OctreeSamples").Device(DEVICE_CPU), OctreeSamplesOp);

}  // namespace tensorflow
