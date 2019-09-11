#include "octree_parser.h"

#include <cuda_runtime.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("OctreeProperty")
    .Input("octree: int8")
    .Attr("property_name: string")
    .Attr("depth: int")
    .Attr("channel: int")
    .Attr("dtype: {int32,float32,uint32}")
    .Output("out_property: dtype")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int channel;
      TF_RETURN_IF_ERROR(c->GetAttr("channel", &channel));
      c->set_output(0, c->MakeShape({channel, c->UnknownDim()}));
      return Status::OK();
    })
    .Doc(R"doc(Octree property operator.)doc");


class OctreePropertyOP : public OpKernel {
 public:
  explicit OctreePropertyOP(OpKernelConstruction* context) :
    OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("property_name", &property_name_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("channel", &channel_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    auto octree_ptr = context->input(0).flat<int8>().data();
    OctreeParser octree;
    octree.set_gpu(octree_ptr);

    const void* property_ptr = nullptr;
    int length = octree.info().node_num(depth_), channel = 1;
    if (property_name_ == "key") {
      property_ptr = octree.key_gpu(depth_);
      channel = octree.info().channel(OctreeInfo::kKey);
      CHECK_EQ(dtype_, DataType::DT_UINT32);
    } else if (property_name_ == "child") {
      property_ptr = octree.children_gpu(depth_);
      channel = octree.info().channel(OctreeInfo::kChild);
      CHECK_EQ(dtype_, DataType::DT_INT32);
    } else if (property_name_ == "neigh") {
      property_ptr = octree.neighbor_gpu(depth_);
      channel = octree.info().channel(OctreeInfo::kNeigh);
      CHECK_EQ(dtype_, DataType::DT_INT32);
    } else if (property_name_ == "feature") {
      property_ptr = octree.feature_gpu(depth_);
      channel = octree.info().channel(OctreeInfo::kFeature);
      CHECK_EQ(dtype_, DataType::DT_FLOAT);
    } else if (property_name_ == "label") {
      property_ptr = octree.label_gpu(depth_);
      channel = octree.info().channel(OctreeInfo::kLabel);
      CHECK_EQ(dtype_, DataType::DT_FLOAT);
    } else if (property_name_ == "split") {
      property_ptr = octree.split_gpu(depth_);
      channel = octree.info().channel(OctreeInfo::kSplit);
      CHECK_EQ(dtype_, DataType::DT_FLOAT);
    } else {
      LOG(FATAL) << "Unsupported Octree Property";
    }
    CHECK_EQ(channel_, channel) << "The specified channel is wrong.";

    Tensor* out_tensor;
    TensorShape out_shape({ channel, length });
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out_tensor));

    int num = channel * length;
    switch (dtype_) {
    case DataType::DT_UINT32: {
      auto ptr = out_tensor->flat<uint32>().data();
      cudaMemcpy(ptr, property_ptr, sizeof(uint32) * num, cudaMemcpyDeviceToDevice);
    }
    break;
    case DataType::DT_INT32: {
      auto ptr = out_tensor->flat<int>().data();
      cudaMemcpy(ptr, property_ptr, sizeof(int) * num, cudaMemcpyDeviceToDevice);
    }
    break;
    case DataType::DT_FLOAT: {
      auto ptr = out_tensor->flat<float>().data();
      cudaMemcpy(ptr, property_ptr, sizeof(float) * num, cudaMemcpyDeviceToDevice);
    }
    break;
    default:
      LOG(FATAL) << "Invalid DataType";
    }
  }

 private:
  string property_name_;
  DataType dtype_;
  int depth_;
  int channel_;  
};

REGISTER_KERNEL_BUILDER(Name("OctreeProperty").Device(DEVICE_GPU), OctreePropertyOP);

}  // namespace tensorflow
