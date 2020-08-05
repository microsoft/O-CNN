#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "points.h"

namespace tensorflow {

REGISTER_OP("PointsNew")
    .Input("pts: float")
    .Input("normals: float")
    .Input("features: float")
    .Input("labels: float")
    .Output("points: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({1}));
      return Status::OK();
    })
    .Doc(R"doc(Create a point cloud.)doc");

REGISTER_OP("PointsSetProperty")
    .Input("points: string")
    .Input("data: float")
    .Attr("property_name: string")
    .Output("points_out: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({1}));
      return Status::OK();
    })
    .Doc(R"doc(Set points property operator.)doc");

class PointsNewOp : public OpKernel {
 public:
  explicit PointsNewOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // input
    auto get_data = [&](vector<float>& vec, int idx) {
      const Tensor& data_in = context->input(idx);
      const int64 num = data_in.NumElements();
      if (num > 0) {
        const float* ptr = data_in.flat<float>().data();
        vec.assign(ptr, ptr + num);
      }
    };
    vector<float> pts, normals, features, labels;
    get_data(pts, 0);
    get_data(normals, 1);
    get_data(features, 2);
    get_data(labels, 3);

    // create the point cloud
    Points point_cloud;
    bool succ = point_cloud.set_points(pts, normals, features, labels);
    CHECK(succ) << "Error occurs when setting points";

    // output
    Tensor* tsr = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{1}, &tsr));
    string& out_str = tsr->flat<string>()(0);
    const vector<char>& points_buf = point_cloud.get_buffer();
    out_str.assign(points_buf.begin(), points_buf.end());
  }
};

class PointsSetPropertyOp : public OpKernel {
 public:
  explicit PointsSetPropertyOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("property_name", &property_name_));
  }

  void Compute(OpKernelContext* context) override {
    // input points
    const Tensor& data_in = context->input(0);
    CHECK_EQ(data_in.NumElements(), 1);
    const Tensor& data = context->input(1);
    CHECK_EQ(data.dims(), 2);
    int num = data.dim_size(0);
    int channel = data.dim_size(1);

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

    // get prop
    PointsInfo::PropType ptype;
    if (property_name_ == "xyz") {
      ptype = PointsInfo::kPoint;
    } else if (property_name_ == "label") {
      ptype = PointsInfo::kLabel;
    } else if (property_name_ == "normal") {
      ptype = PointsInfo::kNormal;
    } else if (property_name_ == "feature") {
      ptype = PointsInfo::kFeature;
    } else {
      LOG(FATAL) << "Unsupported Property: " << property_name_;
    }

    // set the property
    float* ptr = pts.mutable_ptr(ptype);
    CHECK(ptr != nullptr) << "The property does not exist.";
    CHECK_EQ(channel, pts.info().channel(ptype)) << "The channel is wrong.";
    CHECK_EQ(num, pts.info().pt_num()) << "The point number is wrong.";
    memcpy(ptr, data.flat<float>().data(), num * channel * sizeof(float));

    // output
    Tensor* out_data = nullptr;
    const TensorShape& shape = data_in.shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out_data));
    string& out_str = out_data->flat<string>()(0);
    out_str.assign(points_buf.begin(), points_buf.end());
  }

 private:
  string property_name_;
  int channel_;
};

REGISTER_KERNEL_BUILDER(Name("PointsSetProperty").Device(DEVICE_CPU), PointsSetPropertyOp);
REGISTER_KERNEL_BUILDER(Name("PointsNew").Device(DEVICE_CPU), PointsNewOp);

}  // namespace tensorflow
