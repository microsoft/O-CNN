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
.Attr("axis: string='y'")
.Attr("depth: int=6")
.Attr("offset: float=0.55")
.Output("out_points: string")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
})
.Doc(R"doc(Points Database operator.)doc");


class TransformPointsOP : public OpKernel {
 public:
  explicit TransformPointsOP(OpKernelConstruction* context) :
    OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("offset", &offset_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data_in = context->input(0);
    rotate_ = context->input(1).flat<float>()(0);
    scale_  = context->input(2).flat<float>()(0);
    jitter_ = context->input(3).flat<float>()(0);
    CHECK_EQ(data_in.NumElements(), 1);
    CHECK_GE(scale_, 0.1f) << "The scale should be larger than 0.1";

    // copy the data out of the input tensor
    auto points_array = data_in.flat<string>();
    vector<char> points_buf(points_array(0).begin(), points_array(0).end());

    // init the points
    Points point_cloud_;
    point_cloud_.set(points_buf.data());

    // check the points
    string msg;
    bool succ = point_cloud_.info().check_format(msg);
    CHECK(succ) << msg;

    // bounding sphere
    float radius_, center_[3];
    bounding_sphere(radius_, center_, point_cloud_.points(), point_cloud_.info().pt_num());

    // centralize & displacement
    const float kEPS = 1.0e-10f;
    float dis[3] = { -center_[0], -center_[1], -center_[2] };
    point_cloud_.translate(dis);
    if (offset_ > kEPS) {
      float offset = offset_ * 2.0f * radius_ / float(1 << depth_);
      point_cloud_.displace(offset);
      radius_ += offset;
    }

    // data augmentation: rotate the point cloud
    float axis[] = { 0.0f, 0.0f, 0.0f };
    if (axis_ == "x") axis[0] = 1.0f;
    else if (axis_ == "y") axis[1] = 1.0f;
    else axis[2] = 1.0f;
    point_cloud_.rotate(rotate_, axis);

    // jitter. todo: input a float[3] for jittering
    if (fabs(jitter_) > kEPS) {
      float jitter = jitter_ * 2.0f * radius_ / float(1 << depth_);
      dis[0] = dis[1] = dis[2] = jitter;
      point_cloud_.translate(dis);
      radius_ += fabs(jitter);
    }

    // scale and clip the points to the box [-1, 1]^3,
    if (radius_ == 0) radius_ = kEPS;
    point_cloud_.uniform_scale(scale_ / radius_);
    if (scale_ > 1.0f) {
      float bbmin_[] = { -1.0f, -1.0f, -1.0f };
      float bbmax_[] = { 1.0f, 1.0f, 1.0f };
      point_cloud_.clip(bbmin_, bbmax_);
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
  float rotate_;
  float scale_;
  float offset_;
  float jitter_;
};


REGISTER_KERNEL_BUILDER(Name("TransformPoints").Device(DEVICE_CPU), TransformPointsOP);

}  // namespace tensorflow
