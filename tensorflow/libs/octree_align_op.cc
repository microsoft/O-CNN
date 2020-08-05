#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "octree_nn.h"
#include "octree_parser.h"

namespace tensorflow {

REGISTER_OP("OctreeAlign")
    .Input("in_data: float")
    .Input("src_octree: int8")
    .Input("des_octree: int8")
    .Attr("depth: int")
    .Output("out_data: float")
    .Output("key_index: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      auto shape = c->input(0);
      TF_RETURN_IF_ERROR(c->ReplaceDim(shape, 2, c->UnknownDim(), &shape));
      c->set_output(0, shape);
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(Octree align operator.)doc");

REGISTER_OP("OctreeAlignGrad")
    .Input("in_data: float")
    .Input("key_index: int32")
    .Output("out_data: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      auto shape = c->input(0);
      TF_RETURN_IF_ERROR(c->ReplaceDim(shape, 2, c->UnknownDim(), &shape));
      c->set_output(0, shape);
      return Status::OK();
    })
    .Doc(R"doc(Octree align grad operator.)doc");

class OctreeAlignOp : public OpKernel {
 public:
  explicit OctreeAlignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &curr_depth_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& src_data = context->input(0);
    auto src_ptr = src_data.flat<float>().data();
    int src_h = src_data.dim_size(2);
    int channel = src_data.dim_size(1);

    // octrees
    OctreeParser src_octree, des_octree;
    src_octree.set_gpu(context->input(1).flat<int8>().data());
    des_octree.set_gpu(context->input(2).flat<int8>().data());
    int des_h = des_octree.info().node_num(curr_depth_);
    CHECK_EQ(src_octree.info().node_num(curr_depth_), src_h);

    // get key
    const uintk* src_key = src_octree.key_gpu(curr_depth_);
    Tensor src_key_tensor;
    if (src_octree.info().is_key2xyz()) {
      xyz2key_gpu_op(context, &src_key_tensor, src_key, src_h, curr_depth_);
      src_key = src_key_tensor.flat<uintk>().data();
    }
    const uintk* des_key = des_octree.key_gpu(curr_depth_);
    Tensor des_key_tensor;
    if (des_octree.info().is_key2xyz()) {
      xyz2key_gpu_op(context, &des_key_tensor, des_key, des_h, curr_depth_);
      des_key = des_key_tensor.flat<uintk>().data();
    }

    // binary search
    Tensor* idx_tensor = nullptr;
    TensorShape idx_shape({src_h});
    OP_REQUIRES_OK(context, context->allocate_output(1, idx_shape, &idx_tensor));
    auto idx_ptr = idx_tensor->flat<int>().data();
    search_key_gpu(idx_ptr, des_key, des_h, src_key, src_h);

    // out data
    Tensor* des_tensor = nullptr;
    TensorShape des_shape({1, channel, des_h, 1});
    OP_REQUIRES_OK(context, context->allocate_output(0, des_shape, &des_tensor));
    auto des_ptr = des_tensor->flat<float>().data();

    // exec
    align_forward_gpu(des_ptr, des_h, channel, src_ptr, src_h, idx_ptr);
  }

 protected:
  void xyz2key_gpu_op(OpKernelContext* context, Tensor* key_tensor,
                      const uintk* xyz, const int num, const int depth) {
    auto dtype = std::is_same<uintk, uint32>::value ? DT_UINT32 : DT_UINT64;
    OP_REQUIRES_OK(
        context, context->allocate_temp(dtype, TensorShape({num}), key_tensor));
    auto ptr = key_tensor->flat<uintk>().data();
    xyz2key_gpu(ptr, xyz, num, depth);
  }

 private:
  int curr_depth_;
};

class OctreeAlignGradOp : public OpKernel {
 public:
  explicit OctreeAlignGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // gradients
    const Tensor& des_data = context->input(0);
    auto des_ptr = des_data.flat<float>().data();
    int channel = des_data.dim_size(1);
    int des_h = des_data.dim_size(2);

    // index
    const Tensor& idx_tensor = context->input(1);
    int src_h = idx_tensor.dim_size(0);
    auto idx_ptr = idx_tensor.flat<int>().data();

    // grad out
    Tensor* src_tensor = nullptr;
    TensorShape src_shape({1, channel, src_h, 1});
    OP_REQUIRES_OK(context, context->allocate_output(0, src_shape, &src_tensor));
    auto src_ptr = src_tensor->flat<float>().data();

    // exec
    align_backward_gpu(des_ptr, des_h, channel, src_ptr, src_h, idx_ptr);
  }
};

REGISTER_KERNEL_BUILDER(Name("OctreeAlign").Device(DEVICE_GPU), OctreeAlignOp);
REGISTER_KERNEL_BUILDER(Name("OctreeAlignGrad").Device(DEVICE_GPU), OctreeAlignGradOp);

}  // namespace tensorflow
