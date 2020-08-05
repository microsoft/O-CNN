#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "octree_nn.h"
#include "octree_parser.h"

namespace tensorflow {

REGISTER_OP("OctreeBilinear")
    .Input("octree: int8")
    .Attr("curr_depth: int")
    .Attr("target_depth: int")
    .Output("index: int32")
    .Output("fracs: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim(), 8}));
      c->set_output(1, c->MakeShape({c->UnknownDim(), 3}));
      return Status::OK();
    })
    .Doc(R"doc(Octree bilinear operator.)doc");

class OctreeBilinearOp : public OpKernel {
 public:
  explicit OctreeBilinearOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("curr_depth", &curr_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("target_depth", &target_depth_));
    CHECK_GT(curr_depth_, 0)
         << "The curr_depth should be larger than 0";
    CHECK_GT(target_depth_, curr_depth_)
        << "The target_depth should be larger than curr_depth";
  }

  void Compute(OpKernelContext* context) override {
    // octree
    OctreeParser octree_;
    octree_.set_gpu(context->input(0).flat<int8>().data());
    CHECK_LE(target_depth_, octree_.info().depth())
        << "The target_depth should be smaller than the octree depth";

    // get key & xyz
    Tensor src_buffer;
    int src_h = octree_.info().node_num(curr_depth_);
    const uintk* src_key = octree_.key_gpu(curr_depth_);
    if (octree_.info().is_key2xyz()) {
      xyz2key_gpu_op(context, &src_buffer, src_key, src_h, curr_depth_);
      src_key = src_buffer.flat<uintk>().data();
    }

    Tensor des_buffer;
    int des_h = octree_.info().node_num(target_depth_);
    const uintk* des_xyz = octree_.key_gpu(target_depth_);
    if (!octree_.info().is_key2xyz()) {
      key2xyz_gpu_op(context, &des_buffer, des_xyz, des_h, target_depth_);
      des_xyz = des_buffer.flat<uintk>().data();
    }

    // out data
    Tensor* idx_tensor = nullptr;
    TensorShape idx_shape({des_h, 8});
    OP_REQUIRES_OK(context, context->allocate_output(0, idx_shape, &idx_tensor));
    auto idx_ptr = idx_tensor->flat<int>().data();

    Tensor* frac_tensor = nullptr;
    TensorShape frac_shape({des_h, 3});
    OP_REQUIRES_OK(context, context->allocate_output(1, frac_shape, &frac_tensor));
    auto frac_ptr = frac_tensor->flat<float>().data();

    // calc bilinear xyz
    Tensor buf_xyz;
    TensorShape rst_shape({des_h, 8});
    OP_REQUIRES_OK(context, context->allocate_temp(get_dtype(), rst_shape, &buf_xyz));
    auto rst_xyz = buf_xyz.flat<uintk>().data();
    bilinear_xyz_gpu(rst_xyz, frac_ptr, curr_depth_, des_xyz, target_depth_, des_h);

    Tensor buf_key;
    xyz2key_gpu_op(context, &buf_key, rst_xyz, des_h * 8, target_depth_);
    auto rst_key = buf_key.flat<uintk>().data();

    // binary search
    search_key_gpu(idx_ptr, src_key, src_h, rst_key, des_h * 8);
  }

 protected:
  void xyz2key_gpu_op(OpKernelContext* context, Tensor* key_tensor,
      const uintk* xyz, const int num, const int depth) {
    OP_REQUIRES_OK(context, 
        context->allocate_temp(get_dtype(), TensorShape({num}), key_tensor));
    auto ptr = key_tensor->flat<uintk>().data();
    xyz2key_gpu(ptr, xyz, num, depth);
  }

  void key2xyz_gpu_op(OpKernelContext* context, Tensor* xyz_tensor,
      const uintk* key, const int num, const int depth) {
    OP_REQUIRES_OK(context, 
        context->allocate_temp(get_dtype(), TensorShape({num}), xyz_tensor));
    auto ptr = xyz_tensor->flat<uintk>().data();
    key2xyz_gpu(ptr, key, num, depth);
  }

  DataType get_dtype() { return sizeof(uintk) == 4 ? DT_UINT32 : DT_UINT64; }

 private:
  int curr_depth_;
  int target_depth_;
};

REGISTER_KERNEL_BUILDER(Name("OctreeBilinear").Device(DEVICE_GPU), OctreeBilinearOp);

}  // namespace tensorflow
