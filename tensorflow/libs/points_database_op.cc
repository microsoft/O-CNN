#include "octree.h"
#include "math_functions.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>


namespace tensorflow {

REGISTER_OP("PointsDatabase")
    .Input("in_points: string")
    .Input("rotate: float")
    .Attr("axis: string='y'")
    .Attr("depth: int=6")
    .Attr("full_depth: int=2")
    .Attr("offset: float=0.55")
    .Attr("node_dis: bool=False")
    .Attr("node_feature: bool=False")
    .Attr("split_label: bool=False")
    .Attr("adaptive: bool=False")
    .Attr("adp_depth: int=4")
    .Attr("th_normal: float=0.1")
    .Attr("th_distance: float=2.0")
    .Attr("extrapolate: bool=False")
    .Attr("key2xyz: bool=False")
    .Output("out_octree: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Points Database operator.)doc");


class PointsDatabaseOp : public OpKernel {
 public:
  explicit PointsDatabaseOp(OpKernelConstruction* context) :
    OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("full_depth", &full_depth_));
    // OP_REQUIRES_OK(context, context->GetAttr("rotate", &rotate_));
    OP_REQUIRES_OK(context, context->GetAttr("offset", &offset_));
    OP_REQUIRES_OK(context, context->GetAttr("node_dis", &node_dis_));
    OP_REQUIRES_OK(context, context->GetAttr("node_feature", &node_feature_));
    OP_REQUIRES_OK(context, context->GetAttr("split_label", &split_label_));
    OP_REQUIRES_OK(context, context->GetAttr("adaptive", &adaptive_));
    OP_REQUIRES_OK(context, context->GetAttr("adp_depth", &adp_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("th_distance", &th_distance_));
    OP_REQUIRES_OK(context, context->GetAttr("th_normal", &th_normal_));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolate", &extrapolate_));
    OP_REQUIRES_OK(context, context->GetAttr("key2xyz", &key2xyz_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data_in = context->input(0);
    auto points_array = data_in.flat<string>();
    CHECK_EQ(data_in.NumElements(), 1);

    rotate_ = context->input(1).flat<float>()(0);
    
    // copy the data out of the input tensor
    vector<char> buf(points_array(0).begin(), points_array(0).end());

    // init the points
    Points point_cloud_;
    point_cloud_.set(buf.data());

    // check the points
    string msg;
    bool succ = point_cloud_.info().check_format(msg);
    CHECK(succ) << msg;

    // bounding sphere
    float radius_, center_[3];
    bounding_sphere(radius_, center_, point_cloud_.points(), point_cloud_.info().pt_num());

    // centralize & displacement
    float dis[3] = { -center_[0], -center_[1], -center_[2] };
    point_cloud_.translate(dis);
    if (offset_ > 1.0e-10f) {
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

    // init the octree info
    OctreeInfo octree_info_;
    octree_info_.initialize(depth_, full_depth_, node_dis_,
      node_feature_, split_label_, adaptive_, adp_depth_,
      th_distance_, th_normal_, key2xyz_, extrapolate_, false,
      point_cloud_);
    // the point cloud has been centralized,
    // so initializing the bbmin & bbmax in the following way
    float bbmin[] = { -radius_, -radius_, -radius_ };
    float bbmax[] = { radius_, radius_, radius_ };
    octree_info_.set_bbox(bbmin, bbmax);

    // build the octree
    Octree octree_;
    octree_.build(octree_info_, point_cloud_);
    // Modify the bounding box before saving, because the center of
    // the point cloud is translated to (0, 0, 0) when building the octree
    octree_.mutable_info().set_bbox(radius_, center_);
    const vector<char>& octree_buf = octree_.buffer();

    // output
    Tensor* out_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, data_in.shape(), &out_data));
    string& out_str = out_data->flat<string>()(0);
    out_str.assign(octree_buf.begin(), octree_buf.end());
  }

 private:
  string axis_;
  int depth_;
  int full_depth_;
  float rotate_;
  float offset_;
  bool node_dis_;
  bool node_feature_;
  bool split_label_;
  bool adaptive_;
  int adp_depth_;
  float th_distance_;
  float th_normal_;
  bool extrapolate_;
  bool key2xyz_;
};


REGISTER_KERNEL_BUILDER(Name("PointsDatabase").Device(DEVICE_CPU), PointsDatabaseOp);

}  // namespace tensorflow
