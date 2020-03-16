#include "octree_parser.h"
#include "octree_info.h"

#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

namespace tensorflow {

REGISTER_OP("OctreeSetProperty")
    .Input("in_octree: int8")
    .Input("in_property: dtype")
    .Attr("property_name: string")
    .Attr("depth: int")
    .Attr("dtype: {int32,float32,uint32}")
    .Output("out_octree: int8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Octree set property operator.)doc");

class OctreeSetPropertyOp : public OpKernel {
 public:
  explicit OctreeSetPropertyOp(OpKernelConstruction* context) :
    OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("property_name", &property_name_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& in_octree = context->input(0);
    const Tensor& in_property = context->input(1);
    auto in_ptr = in_octree.flat<int8>().data();

    Tensor* out_octree;
    OP_REQUIRES_OK(context, context->allocate_output(0, in_octree.shape(), &out_octree));
    auto out_ptr = out_octree->flat<int8>().data();
    cudaMemcpy(out_ptr, in_ptr, in_octree.NumElements(), cudaMemcpyDeviceToDevice);

    OctreeParser oct_parser;
    int length;
    void* property_ptr = nullptr;
    oct_parser.set_gpu(out_ptr);
    length = oct_parser.info().node_num(depth_);
    if (property_name_ == "key") {
      property_ptr = oct_parser.mutable_key_gpu(depth_);
      CHECK_EQ(dtype_, DataType::DT_UINT32);
    } else if (property_name_ == "child") {
      property_ptr = oct_parser.mutable_children_gpu(depth_);
      length *= oct_parser.info().channel(OctreeInfo::kChild);
      CHECK_EQ(dtype_, DataType::DT_INT32);
    } else if (property_name_ == "neigh") {
      property_ptr = oct_parser.mutable_neighbor_gpu(depth_);
      length *= oct_parser.info().channel(OctreeInfo::kNeigh);
      CHECK_EQ(dtype_, DataType::DT_INT32);
    } else if (property_name_ == "feature") {
      property_ptr = oct_parser.mutable_feature_gpu(depth_);
      length *= oct_parser.info().channel(OctreeInfo::kFeature);
      CHECK_EQ(dtype_, DataType::DT_FLOAT);
    } else if (property_name_ == "label") {
      property_ptr = oct_parser.mutable_label_gpu(depth_);
      length *= oct_parser.info().channel(OctreeInfo::kLabel);
      CHECK_EQ(dtype_, DataType::DT_FLOAT);
    } else if (property_name_ == "split") {
      property_ptr = oct_parser.mutable_split_gpu(depth_);
      length *= oct_parser.info().channel(OctreeInfo::kSplit);
      CHECK_EQ(dtype_, DataType::DT_FLOAT);
    } else {
      LOG(FATAL) << "Unsupported Octree Property";
    }
    CHECK_EQ(length, in_property.NumElements()) << "Wrong Property Size";
    switch (dtype_) {
    case DataType::DT_UINT32: {
      auto in_property_ptr = in_property.flat<uint32>().data();
      cudaMemcpy(property_ptr, in_property_ptr, sizeof(uint32) * length,
          cudaMemcpyDeviceToDevice);
    }
    break;
    case DataType::DT_INT32: {
      auto in_property_ptr = in_property.flat<int>().data();
      cudaMemcpy(property_ptr, in_property_ptr, sizeof(int) * length,
          cudaMemcpyDeviceToDevice);
    }
    break;
    case DataType::DT_FLOAT: {
      auto in_property_ptr = in_property.flat<float>().data();
      cudaMemcpy(property_ptr, in_property_ptr, sizeof(float) * length,
          cudaMemcpyDeviceToDevice);
    }
    break;
    default:
      LOG(FATAL) << "Wrong DataType";
    }
  }

 private:
  string property_name_;
  DataType dtype_;
  int depth_;
  
};

REGISTER_KERNEL_BUILDER(Name("OctreeSetProperty").Device(DEVICE_GPU), OctreeSetPropertyOp);

}  // namespace tensorflow
