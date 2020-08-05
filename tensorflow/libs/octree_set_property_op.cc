#include <cuda_runtime.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "octree_info.h"
#include "octree_parser.h"

namespace tensorflow {

REGISTER_OP("OctreeSetProperty")
    .Input("octree_in: int8")
    .Input("property_in: dtype")
    .Attr("property_name: string")
    .Attr("depth: int")
    .Attr("dtype: {int32,float32,uint32,uint64}")
    .Output("octree_out: int8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Octree set property operator.)doc");

class OctreeSetPropertyOp : public OpKernel {
 public:
  explicit OctreeSetPropertyOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("property_name", &property_name_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& octree_in = context->input(0);
    const Tensor& property_in = context->input(1);
    auto ptr_in = octree_in.flat<int8>().data();

    Tensor* octree_out;
    OP_REQUIRES_OK(context, context->allocate_output(0, octree_in.shape(), &octree_out));
    auto ptr_out = octree_out->flat<int8>().data();
    cudaMemcpy(ptr_out, ptr_in, octree_in.NumElements(), cudaMemcpyDeviceToDevice);

    OctreeParser oct_parser;
    oct_parser.set_gpu(ptr_out);
    void* property_ptr = nullptr;
    int length = oct_parser.info().node_num(depth_);
    if (property_name_ == "key") {
      bool key32 = std::is_same<uintk, uint32>::value;
      DataType key_dtype = key32 ? DataType::DT_UINT32 : DataType::DT_UINT64;
      property_ptr = oct_parser.mutable_key_gpu(depth_);
      length *= oct_parser.info().channel(OctreeInfo::kKey);
      CHECK_EQ(dtype_, key_dtype);
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

    CHECK_EQ(length, property_in.NumElements()) << "Wrong Property Size";
    switch (dtype_) {
      case DataType::DT_UINT32: {
        auto property_in_ptr = property_in.flat<uint32>().data();
        cudaMemcpy(property_ptr, property_in_ptr, sizeof(uint32) * length,
                   cudaMemcpyDeviceToDevice);
      } break;
      case DataType::DT_UINT64: {
        auto property_in_ptr = property_in.flat<uint64>().data();
        cudaMemcpy(property_ptr, property_in_ptr, sizeof(uint64) * length,
                   cudaMemcpyDeviceToDevice);
      } break;
      case DataType::DT_INT32: {
        auto property_in_ptr = property_in.flat<int>().data();
        cudaMemcpy(property_ptr, property_in_ptr, sizeof(int) * length,
                   cudaMemcpyDeviceToDevice);
      } break;
      case DataType::DT_FLOAT: {
        auto property_in_ptr = property_in.flat<float>().data();
        cudaMemcpy(property_ptr, property_in_ptr, sizeof(float) * length,
                   cudaMemcpyDeviceToDevice);
      } break;
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
