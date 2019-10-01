#include "octree.h"
#include "math_functions.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>


namespace tensorflow {

REGISTER_OP("TransformPoints")
    .Input("in_points: string")
    .Input("rotate: float")
    .Input("scale: float")
    .Input("jitter: float")
    .Input("radius: float")
    .Input("center: float")
    .Attr("axis: string='y'")
    .Attr("depth: int=6")
    .Attr("offset: float=0.55")
    .Output("out_points: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Points Database operator.)doc");

REGISTER_OP("BoundingSphere")
    .Input("points: string")
    .Output("radius: float")
    .Output("center: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({1}));
      c->set_output(1, c->MakeShape({3}));
      return Status::OK();
    })
    .Doc(R"doc(Compute the bounding sphere of a point cloud.)doc");


class TransformPointsOP : public OpKernel {
 public:
  explicit TransformPointsOP(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("offset", &offset_));
  }

  void Compute(OpKernelContext* context) override {
    // input
    const Tensor& data_in = context->input(0);
    CHECK_EQ(data_in.NumElements(), 1);
    float rotate = context->input(1).flat<float>()(0);
    float scale  = context->input(2).flat<float>()(0);
    CHECK_GE(scale, 0.1f) << "The scale should be larger than 0.1";
    float jitter = context->input(3).flat<float>()(0);
    float radius = context->input(4).flat<float>()(0);
    float center[3] = {0};
    const Tensor& center_in = context->input(5);
    CHECK_EQ(center_in.NumElements(), 3);
    for (int i = 0; i < 3; ++i) {
      center[i] = center_in.flat<float>()(i);
    }

    // copy the data out of the input tensor
    auto points_array = data_in.flat<string>();
    vector<char> points_buf(points_array(0).begin(), points_array(0).end());

    // init the points
    Points pts;
    pts.set(points_buf.data());

    // check the points
    string msg;
    bool succ = pts.info().check_format(msg);
    CHECK(succ) << msg;

    // bounding sphere
    //bounding_sphere(radius, center, pts.points(), pts.info().pt_num());

    // centralize & displacement
    const float kEPS = 1.0e-10f;
    float dis[3] = { -center[0], -center[1], -center[2] };
    pts.translate(dis);
    if (offset_ > kEPS) {
      float offset = offset_ * 2.0f * radius / float(1 << depth_);
      pts.displace(offset);
      radius += offset;
    }

    // data augmentation: rotate the point cloud
    if (rotate > kEPS) {
      float axes[] = { 0.0f, 0.0f, 0.0f };
      if (axis_ == "x") axes[0] = 1.0f;
      else if (axis_ == "y") axes[1] = 1.0f;
      else axes[2] = 1.0f;
      pts.rotate(rotate, axes);
    }

    // jitter. todo: input a float[3] for jittering
    if (fabs(jitter) > kEPS) {
      float jitter = jitter * 2.0f * radius / float(1 << depth_);
      dis[0] = dis[1] = dis[2] = jitter;
      pts.translate(dis);
      radius += fabs(jitter);
    }

    // scale and clip the points to the box [-1, 1]^3,
    if (radius == 0) radius = kEPS;
    pts.uniform_scale(scale / radius);
    if (scale > 1.0f) {
      const float bbmin[] = { -1.0f, -1.0f, -1.0f };
      const float bbmax[] = { 1.0f, 1.0f, 1.0f };
      pts.clip(bbmin, bbmax);
    }

    // output
    Tensor* out_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, data_in.shape(), &out_data));
    string& out_str = out_data->flat<string>()(0);
    out_str.assign(points_buf.begin(), points_buf.end());
  }

 private:
  string axis_;
  int depth_;
  float offset_;
};

class BoundingSphereOP : public OpKernel {
 public:
  explicit BoundingSphereOP(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& data_in = context->input(0);
    CHECK_EQ(data_in.NumElements(), 1);

    // init the points
    Points pts;
    pts.set(data_in.flat<string>()(0).data());

    // check the points
    string msg;
    bool succ = pts.info().check_format(msg);
    CHECK(succ) << msg;

    // bounding sphere
    float radius, center[3];
    bounding_sphere(radius, center, pts.points(), pts.info().pt_num());

    // output
    Tensor* out0 = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({ 1 }), &out0));
    float* ptr0 = out0->flat<float>().data();
    ptr0[0] = radius;

    Tensor* out1 = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({ 3 }), &out1));
    float* ptr1 = out1->flat<float>().data();
    for (int i = 0; i < 3; ++i) { ptr1[i] = center[i]; }
  }
};


REGISTER_KERNEL_BUILDER(Name("TransformPoints").Device(DEVICE_CPU), TransformPointsOP);
REGISTER_KERNEL_BUILDER(Name("BoundingSphere").Device(DEVICE_CPU), BoundingSphereOP);

}  // namespace tensorflow
