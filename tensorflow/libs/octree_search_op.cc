#include "octree_nn.h"
#include "octree_parser.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("OctreeSearch")
    .Input("xyz: float")    // C * H
    .Input("octree: int8")
    .Attr("depth: int")
    //.Attr("dtype: {int32,float32,uint32}")
    .Output("kidx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({ c->UnknownDim() }));
      return Status::OK();
    })
    .Doc(R"doc(Octree search operator.)doc");


class OctreeSearchOp : public OpKernel {
 public:
  explicit OctreeSearchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    //OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    // input
    const Tensor& src_data = context->input(0);
    auto src_ptr = src_data.flat<float>().data();
    int channel = src_data.dim_size(0);
    int src_h = src_data.dim_size(1);
    CHECK_EQ(channel, 4) << "The input channel must be 4: x, y, z, id";

    // coord2xyz
    Tensor src_xyz_tensor;
    coord2xyz_gpu_op(context, &src_xyz_tensor, src_ptr, src_h, channel);
    const uint32* src_xyz = src_xyz_tensor.flat<uint32>().data();

    // xyz2key
    Tensor src_key_tensor;
    xyz2key_gpu_op(context, &src_key_tensor, src_xyz, src_h, depth_);
    const uint32* src_key = src_key_tensor.flat<uint32>().data();

    // octree
    OctreeParser octree_;
    octree_.set_gpu(context->input(1).flat<int8>().data());
    int des_h = octree_.info().node_num(depth_);
    const uint32* des_key = octree_.key_gpu(depth_);
    Tensor des_key_tensor;
    if (octree_.info().is_key2xyz()) {
      xyz2key_gpu_op(context, &des_key_tensor, des_key, des_h, depth_);
      des_key = des_key_tensor.flat<uint32>().data();
    }

    // output
    Tensor* des_tensor = nullptr;
    TensorShape des_shape({ src_h });
    OP_REQUIRES_OK(context, context->allocate_output(0, des_shape, &des_tensor));
    auto idx_ptr = des_tensor->flat<int>().data();

    // binary search
    search_key_gpu(idx_ptr, des_key, des_h, src_key, src_h);
  }

 protected:
  // todo: isolated the following functions out of this file
  void xyz2key_gpu_op(OpKernelContext* context, Tensor* key_tensor,
      const uint32* xyz, const int num, const int depth) {
    OP_REQUIRES_OK(context,
        context->allocate_temp(DT_UINT32, TensorShape({ num }), key_tensor));
    auto ptr = key_tensor->flat<uint32>().data();
    xyz2key_gpu(ptr, xyz, num, depth);
  }

  void coord2xyz_gpu_op(OpKernelContext* context, Tensor* xyz_tensor,
      const float* coord, const int num, const int channel) {
    OP_REQUIRES_OK(context,
        context->allocate_temp(DT_UINT32, TensorShape({ num }), xyz_tensor));
    auto xyz = xyz_tensor->flat<uint32>().data();
    coord2xyz_gpu(xyz, coord, num, channel);
  }

 private:
  int depth_;
  //DataType dtype_;
};


REGISTER_KERNEL_BUILDER(Name("OctreeSearch").Device(DEVICE_GPU), OctreeSearchOp);

}  // namespace tensorflow
