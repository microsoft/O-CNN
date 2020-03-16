#include "octree_nn.h"
#include "octree_parser.h"

#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("OctreeEncodeKey")
    .Input("xyz: uint8")
    .Output("key: uint32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0)}));
      return Status::OK();
    })
    .Doc(R"doc(Encode the (x, y, z, id) to key in uint32)")doc");

REGISTER_OP("OctreeDecodeKey")
    .Input("key: uint32")
    .Output("xyz: uint8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0), 4}));
      return Status::OK();
    })
    .Doc(R"doc(Decode the key to (x, y, z, id) in uint8)")doc");

REGISTER_OP("OctreeKeyToXyz")
    .Input("key: uint32")
    .Attr("depth: int = 8")
    .Output("xyz: uint32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Convert the key to xyz)")doc");

REGISTER_OP("OctreeXyzToKey")
    .Input("xyz: uint32")
    .Attr("depth: int = 8")
    .Output("key: uint32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Convert the xyz to key)")doc");

REGISTER_OP("OctreeSearchKey")
    .Input("key: uint32")
    .Input("octree: int8")
    .Attr("depth: int")
    .Attr("is_xyz: bool = True")
    .Output("kidx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({ c->UnknownDim() }));
      return Status::OK();
    })
    .Doc(R"doc(Octree search operator.)doc");


class OctreeEncodeKeyOp : public OpKernel {
 public:
  explicit OctreeEncodeKeyOp(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& data_in = context->input(0);
    auto ptr_in = data_in.flat<uint8>().data();
    int num = data_in.dim_size(0);
    int channel = data_in.dim_size(1);
    CHECK_EQ(data_in.dims(), 2) << "The dim of input tensor must be 2.";
    CHECK_EQ(channel, 4) << "The channel of input tensor must be 4.";

    // out data
    Tensor* data_out = nullptr;
    TensorShape shape_out({ num });
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_out, &data_out));
    auto ptr_out = data_out->flat<uint32>().data();

    // copy data
    cudaMemcpy(ptr_out, ptr_in, sizeof(uint8) * channel * num, cudaMemcpyDeviceToDevice);
  }
};

class OctreeDecodeKeyOp : public OpKernel {
 public:
  explicit OctreeDecodeKeyOp(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& data_in = context->input(0);
    auto ptr_in = data_in.flat<uint32>().data();
    int num = data_in.dim_size(0);
    CHECK_EQ(data_in.dims(), 1) << "The dim of input tensor must be 1.";

    // out data
    Tensor* data_out = nullptr;
    TensorShape shape_out({num, 4});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_out, &data_out));
    auto ptr_out = data_out->flat<uint8>().data();

    // copy data
    cudaMemcpy(ptr_out, ptr_in, sizeof(uint32) * num, cudaMemcpyDeviceToDevice);
  }
};

class OctreeKeyToXyzOp : public OpKernel {
 public:
  explicit OctreeKeyToXyzOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& data_in = context->input(0);
    const TensorShape& shape_in = data_in.shape();
    auto ptr_in = data_in.flat<uint32>().data();
    int num = shape_in.num_elements();
    CHECK_GE(num, 1) << "The element number of input tensor must be 1.";

    // out data
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_in, &data_out));
    auto ptr_out = data_out->flat<uint32>().data();

    // convert
    key2xyz_gpu(ptr_out, ptr_in, num, depth_);
  }

 private:
  int depth_;
};

class OctreeXyzToKeyOp : public OpKernel {
 public:
  explicit OctreeXyzToKeyOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& data_in = context->input(0);
    const TensorShape& shape_in = data_in.shape();
    auto ptr_in = data_in.flat<uint32>().data();
    int num = shape_in.num_elements();
    CHECK_GE(num, 1) << "The element number of input tensor must be 1.";

    // out data
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_in, &data_out));
    auto ptr_out = data_out->flat<uint32>().data();

    // convert
    xyz2key_gpu(ptr_out, ptr_in, num, depth_);
  }

 private:
  int depth_;
};

class OctreeSearchKeyOp : public OpKernel {
 public:
  explicit OctreeSearchKeyOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("is_xyz", &is_xyz_));
  }

  void Compute(OpKernelContext* context) override {
    // input
    const Tensor& data_in = context->input(0);
    const TensorShape& shape_in = data_in.shape();
    const uint32* src_key = data_in.flat<uint32>().data();
    int src_h = shape_in.num_elements();
    CHECK_GE(src_h, 1) << "The element number of input tensor must be 1.";

    // xyz2key
    Tensor src_key_tensor;
    if (is_xyz_) {
      xyz2key_gpu_op(context, &src_key_tensor, src_key, src_h, depth_);
      src_key = src_key_tensor.flat<uint32>().data();
    }

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
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_in, &des_tensor));
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

 private:
  int depth_;
  bool is_xyz_;
};


REGISTER_KERNEL_BUILDER(Name("OctreeDecodeKey").Device(DEVICE_GPU), OctreeDecodeKeyOp);
REGISTER_KERNEL_BUILDER(Name("OctreeEncodeKey").Device(DEVICE_GPU), OctreeEncodeKeyOp);
REGISTER_KERNEL_BUILDER(Name("OctreeKeyToXyz").Device(DEVICE_GPU), OctreeKeyToXyzOp);
REGISTER_KERNEL_BUILDER(Name("OctreeXyzToKey").Device(DEVICE_GPU), OctreeXyzToKeyOp);
REGISTER_KERNEL_BUILDER(Name("OctreeSearchKey").Device(DEVICE_GPU), OctreeSearchKeyOp);

}  // namespace tensorflow
