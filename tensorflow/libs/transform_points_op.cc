#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "math_functions.h"
#include "points.h"
#include "transform_points.h"

namespace tensorflow {

REGISTER_OP("TransformPoints")
    .Input("points: string")
    .Input("angle: float")
    .Input("scale: float")
    .Input("jitter: float")
    .Input("radius: float")
    .Input("center: float")
    .Input("ratio: float")
    .Input("dim: int32")
    .Input("stddev: float")
    .Attr("axis: string='y'")  // todo: delete this attribute
    .Attr("depth: int=6")
    .Attr("offset: float=0.55")
    .Output("points_out: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Transform points.)doc");

REGISTER_OP("NormalizePoints")
    .Input("points: string")
    .Input("radius: float")
    .Input("center: float")
    .Output("points_out: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Move the center to the origin, and normalize input to [-1, 1].)doc");

REGISTER_OP("BoundingSphere")
    .Input("points: string")
    .Attr("method: string='sphere'")
    .Output("radius: float")
    .Output("center: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({1}));
      c->set_output(1, c->MakeShape({3}));
      return Status::OK();
    })
    .Doc(R"doc(Compute the bounding sphere of a point cloud.)doc");

class TransformPointsOp : public OpKernel {
 public:
  explicit TransformPointsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("offset", &offset_));
  }

  void Compute(OpKernelContext* context) override {
    // input
    auto extract_param = [](float* vec, const Tensor& ts) {
      for (int i = 0; i < 3 && i < ts.NumElements(); ++i) {
        vec[i] = ts.flat<float>()(i);
      }
    };
    const Tensor& data_in = context->input(0);
    // float rotate = context->input(1).flat<float>()(0);
    float angle[3] = {0};
    extract_param(angle, context->input(1));
    float scales[3] = {1.0f, 1.0f, 1.0f};
    extract_param(scales, context->input(2));
    float jitter[3] = {0};
    extract_param(jitter, context->input(3));
    float radius = context->input(4).flat<float>()(0);
    float center[3] = {0};
    extract_param(center, context->input(5));
    float ratio = context->input(6).flat<float>()(0);
    int dim = context->input(7).flat<int>()(0);
    float stddev[3] = {0};  // std_points, std_normals, std_features
    extract_param(stddev, context->input(8));

    // check
    CHECK_EQ(data_in.NumElements(), 1);
    for (int i = 0; i < 3; ++i) {
      CHECK_GE(scales[i], 0.1f) << "The scale should be larger than 0.1";
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

    // centralize
    float dis[3] = {-center[0], -center[1], -center[2]};
    if (dis[0] != 0.0f || dis[1] != 0.0f || dis[2] != 0.0f) {
      pts.translate(dis);
    }

    // displacement
    const float kEPS = 1.0e-10f;
    if (offset_ > kEPS) {
      // !!! rescale the offset, relative to the octree node size
      float offset = offset_ * 2.0f * radius / float(1 << depth_);
      pts.displace(offset);
      radius += offset;
    }

    // data augmentation: rotate the point cloud
    if (fabs(angle[0]) > kEPS || fabs(angle[1]) > kEPS || fabs(angle[2]) > kEPS) {
      if (axis_ == "x") {
        angle[1] = angle[2] = 0;
      } else if (axis_ == "y") {
        angle[0] = angle[2] = 0;
      } else if (axis_ == "z") {
        angle[0] = angle[1] = 0;
      } else {
      }
      pts.rotate(angle);
    }

    // jitter
    float max_jitter = -1.0;
    for (int i = 0; i < 3; i++) {
      // !!! rescale the jitter, relative to the radius
      jitter[i] *= 2.0 * radius;
      if (max_jitter < fabs(jitter[i])) {
        max_jitter = fabs(jitter[i]);
      }
    }
    if (fabs(max_jitter) > kEPS) {
      pts.translate(jitter);
      // radius += max_jitter;
    }

    // scale to [-1, 1]^3
    if (radius == 0) radius = kEPS;
    for (int i = 0; i < 3; ++i) { scales[i] /= radius; }
    if (scales[0] != 1.0f || scales[1] != 1.0f || scales[2] != 1.0f) {
      pts.scale(scales);
    }

    // add noise
    if (stddev[0] > 0 || stddev[1] > 0 || stddev[2] > 0) {
      pts.add_noise(stddev[0], stddev[1]);
    }

    // clip the points to the box[-1, 1] ^ 3,
    const float bbmin[] = {-1.0f, -1.0f, -1.0f};
    const float bbmax[] = {1.0f,   1.0f,  1.0f};
    pts.clip(bbmin, bbmax);
    // float max_scale = -1.0f;
    // for (int i = 0; i < 3; ++i) {
    //   if (max_scale < scales[i]) { max_scale = scales[i]; }
    // }
    // if (max_scale > 1.0f || max_jitter > kEPS) {
    //   pts.clip(bbmin, bbmax);
    // }

    // dropout points
    if (dim > 0 && ratio > 0) {
      DropPoints drop_points(dim, ratio, bbmin, bbmax);
      drop_points.dropout(pts);
    }

    // output
    Tensor* out_data = nullptr;
    const TensorShape& shape = data_in.shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out_data));
    string& out_str = out_data->flat<string>()(0);
    out_str.assign(points_buf.begin(), points_buf.end());
  }

 private:
  string axis_;
  int depth_;
  float offset_;
};

class NormalizePointsOp : public OpKernel {
 public:
  explicit NormalizePointsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // input
    const string& data_in = context->input(0).flat<string>()(0);
    const float radius = context->input(1).flat<float>()(0);
    const float* center = context->input(2).flat<float>().data();

    // check
    CHECK_EQ(context->input(0).NumElements(), 1);
    CHECK_EQ(context->input(1).NumElements(), 1);
    CHECK_EQ(context->input(2).NumElements(), 3);

    // copy the data out of the input tensor
    vector<char> points_buf(data_in.begin(), data_in.end());

    // init the points
    Points pts;
    pts.set(points_buf.data());

    // check the points
    string msg;
    bool succ = pts.info().check_format(msg);
    CHECK(succ) << msg;

    // centralize
    const float dis[3] = {-center[0], -center[1], -center[2]};
    if (dis[0] != 0.0f || dis[1] != 0.0f || dis[2] != 0.0f) {
      pts.translate(dis);
    }

    // scale to [-1, 1]
    CHECK_GE(radius, 0.0f);
    const float inv_radius = 1.0f / radius;
    const float scales[3] = {inv_radius, inv_radius, inv_radius};
    if (scales[0] != 1.0f || scales[1] != 1.0f || scales[2] != 1.0f) {
      pts.scale(scales);
    }

    // output
    Tensor* out_data = nullptr;
    const TensorShape& shape = context->input(0).shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out_data));
    string& out_str = out_data->flat<string>()(0);
    out_str.assign(points_buf.begin(), points_buf.end());
  }
};

class BoundingSphereOp : public OpKernel {
 public:
  explicit BoundingSphereOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
  }

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
    float radius = 0.0f, center[3] = {0.0f};
    if (method_ == "sphere") {
      bounding_sphere(radius, center, pts.points(), pts.info().pt_num());
    } else {
      float bbmin[3] = {0.0f}, bbmax[3] = {0.0f};
      bounding_box(bbmin, bbmax, pts.points(), pts.info().pt_num());
      for (int j = 0; j < 3; ++j) {
        center[j] = (bbmax[j] + bbmin[j]) / 2.0f;
        float width = (bbmax[j] - bbmin[j]) / 2.0f;
        radius += width * width;
      }
      radius = sqrtf(radius + 1.0e-20f);
    }

    // output
    Tensor* out0 = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1}), &out0));
    float* ptr0 = out0->flat<float>().data();
    ptr0[0] = radius;

    Tensor* out1 = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({3}), &out1));
    float* ptr1 = out1->flat<float>().data();
    for (int i = 0; i < 3; ++i) { ptr1[i] = center[i]; }
  }

 private:
  string method_;
};

REGISTER_KERNEL_BUILDER(Name("TransformPoints").Device(DEVICE_CPU), TransformPointsOp);
REGISTER_KERNEL_BUILDER(Name("NormalizePoints").Device(DEVICE_CPU), NormalizePointsOp);
REGISTER_KERNEL_BUILDER(Name("BoundingSphere").Device(DEVICE_CPU), BoundingSphereOp);

}  // namespace tensorflow
