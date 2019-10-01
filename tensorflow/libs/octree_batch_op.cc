#include "merge_octrees.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>


namespace tensorflow {

REGISTER_OP("OctreeBatch")
    .Input("batch_data: string")
    .Output("octree: int8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(1));
      return Status::OK();
    })
    .Doc(R"doc(Merge a batch of octrees.)doc");


class OctreeBatchOp : public OpKernel {
 public:
  explicit OctreeBatchOp(OpKernelConstruction* context) 
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // input octrees
    const Tensor& data_in = context->input(0);
    auto octree_buffer = data_in.flat<string>();
    // int batch_size = data_in.shape().dim_size(0);
    int batch_size = data_in.shape().num_elements();
    vector<const char*> octrees_in;
    for (int i = 0; i < batch_size; ++i) {
      octrees_in.push_back(octree_buffer(i).data());
    }

    // merge octrees
    vector<char> octree_out;
    merge_octrees(octree_out, octrees_in);

    // copy output
    Tensor* out_data = nullptr;
    TensorShape out_shape({ (long long int) octree_out.size() });
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out_data));
    auto out_ptr = out_data->flat<int8>().data();
    memcpy(out_ptr, octree_out.data(), octree_out.size());
  }
};


REGISTER_KERNEL_BUILDER(Name("OctreeBatch").Device(DEVICE_CPU), OctreeBatchOp);

} // namespace tensorflow
