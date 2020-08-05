#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <type_traits>
#include "octree_nn.h"
#include "octree_parser.h"

namespace tensorflow {

inline string uintk_modifier(string str) {
  string dtype = std::is_same<uintk, uint32>::value ? "uint32" : "uint64";
  size_t pos = str.find("uintk");
  return str.replace(pos, 5, dtype);
}

inline string uints_modifier(string str) {
  string dtype = std::is_same<uintk, uint32>::value ? "uint8" : "uint16";
  size_t pos = str.find("uints");
  return str.replace(pos, 5, dtype);
}

REGISTER_OP("OctreeEncodeKey")
    .Input(uints_modifier("xyz: uints"))
    .Output(uintk_modifier("key: uintk"))
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0)}));
      return Status::OK();
    })
    .Doc(R"doc(Encode the (x, y, z, id) to key in uintk)")doc");

REGISTER_OP("OctreeDecodeKey")
    .Input(uintk_modifier("key: uintk"))
    .Output(uints_modifier("xyz: uints"))
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0), 4}));
      return Status::OK();
    })
    .Doc(R"doc(Decode the key to (x, y, z, id) in uints)")doc");

REGISTER_OP("OctreeKeyToXyz")
    .Input(uintk_modifier("key: uintk"))
    .Attr("depth: int = 8")
    .Output(uintk_modifier("xyz: uintk"))
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Convert the key to xyz)")doc");

REGISTER_OP("OctreeXyzToKey")
    .Input(uintk_modifier("xyz: uintk"))
    .Attr("depth: int = 8")
    .Output(uintk_modifier("key: uintk"))
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Convert the xyz to key)")doc");

REGISTER_OP("OctreeSearchKey")
    .Input(uintk_modifier("key: uintk"))
    .Input("octree: int8")
    .Attr("depth: int")
    .Attr("is_xyz: bool = True")
    .Output("kidx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(Octree search operator.)doc");

class OctreeEncodeKeyOp : public OpKernel {
 public:
  explicit OctreeEncodeKeyOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef typename KeyTrait<uintk>::uints uints;

    // in data
    const Tensor& data_in = context->input(0);
    auto ptr_in = data_in.flat<uints>().data();
    int num = data_in.dim_size(0);
    int channel = data_in.dim_size(1);
    CHECK_EQ(data_in.dims(), 2) << "The dim of input tensor must be 2.";
    CHECK_EQ(channel, 4) << "The channel of input tensor must be 4.";

    // out data
    Tensor* data_out = nullptr;
    TensorShape shape_out({num});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_out, &data_out));
    auto ptr_out = data_out->flat<uintk>().data();

    // copy data
    int n = channel * num;
    cudaMemcpy(ptr_out, ptr_in, sizeof(uints) * n, cudaMemcpyDeviceToDevice);
  }
};

class OctreeDecodeKeyOp : public OpKernel {
 public:
  explicit OctreeDecodeKeyOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef typename KeyTrait<uintk>::uints uints;

    // in data
    const Tensor& data_in = context->input(0);
    auto ptr_in = data_in.flat<uintk>().data();
    int num = data_in.dim_size(0);
    CHECK_EQ(data_in.dims(), 1) << "The dim of input tensor must be 1.";

    // out data
    Tensor* data_out = nullptr;
    TensorShape shape_out({num, 4});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_out, &data_out));
    auto ptr_out = data_out->flat<uints>().data();

    // copy data
    cudaMemcpy(ptr_out, ptr_in, sizeof(uintk) * num, cudaMemcpyDeviceToDevice);
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
    auto ptr_in = data_in.flat<uintk>().data();
    int num = shape_in.num_elements();
    CHECK_GE(num, 1) << "The element number of input tensor must be 1.";

    // out data
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_in, &data_out));
    auto ptr_out = data_out->flat<uintk>().data();

    // convert
    key2xyz_gpu(ptr_out, ptr_in, num, depth_);
  }

 private:
  int depth_;
};

class OctreeXyzToKeyOp : public OpKernel {
 public:
  explicit OctreeXyzToKeyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& data_in = context->input(0);
    const TensorShape& shape_in = data_in.shape();
    auto ptr_in = data_in.flat<uintk>().data();
    int num = shape_in.num_elements();
    CHECK_GE(num, 1) << "The element number of input tensor must be 1.";

    // out data
    Tensor* data_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_in, &data_out));
    auto ptr_out = data_out->flat<uintk>().data();

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
    const uintk* src_key = data_in.flat<uintk>().data();
    int src_h = shape_in.num_elements();
    CHECK_GE(src_h, 1) << "The element number of input tensor must be 1.";

    // xyz2key
    Tensor src_key_tensor;
    if (is_xyz_) {
      xyz2key_gpu_op(context, &src_key_tensor, src_key, src_h, depth_);
      src_key = src_key_tensor.flat<uintk>().data();
    }

    // octree
    OctreeParser octree_;
    octree_.set_gpu(context->input(1).flat<int8>().data());
    int des_h = octree_.info().node_num(depth_);
    const uintk* des_key = octree_.key_gpu(depth_);
    Tensor des_key_tensor;
    if (octree_.info().is_key2xyz()) {
      xyz2key_gpu_op(context, &des_key_tensor, des_key, des_h, depth_);
      des_key = des_key_tensor.flat<uintk>().data();
    }

    // output
    Tensor* des_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape_in, &des_tensor));
    auto idx_ptr = des_tensor->flat<int>().data();

    // binary search
    search_key_gpu(idx_ptr, des_key, des_h, src_key, src_h);
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
  int depth_;
  bool is_xyz_;
};

REGISTER_KERNEL_BUILDER(Name("OctreeDecodeKey").Device(DEVICE_GPU), OctreeDecodeKeyOp);
REGISTER_KERNEL_BUILDER(Name("OctreeEncodeKey").Device(DEVICE_GPU), OctreeEncodeKeyOp);
REGISTER_KERNEL_BUILDER(Name("OctreeKeyToXyz").Device(DEVICE_GPU), OctreeKeyToXyzOp);
REGISTER_KERNEL_BUILDER(Name("OctreeXyzToKey").Device(DEVICE_GPU), OctreeXyzToKeyOp);
REGISTER_KERNEL_BUILDER(Name("OctreeSearchKey").Device(DEVICE_GPU), OctreeSearchKeyOp);

}  // namespace tensorflow
