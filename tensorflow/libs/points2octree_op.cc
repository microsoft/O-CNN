#include "octree.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>


namespace tensorflow {

REGISTER_OP("PointsToOctree")
    .Input("in_points: string")
    .Attr("depth: int=6")
    .Attr("full_depth: int=2")
    .Attr("node_dis: bool=False")
    .Attr("node_feature: bool=False")
    .Attr("split_label: bool=False")
    .Attr("adaptive: bool=False")
    .Attr("adp_depth: int=4")
    .Attr("th_normal: float=0.1")
    .Attr("th_distance: float=2.0")
    .Attr("extrapolate: bool=False")
    .Attr("save_pts: bool=False")
    .Attr("key2xyz: bool=False")
    .Output("out_octree: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(Points To Octree operator.)doc");


class PointsToOctreeOp : public OpKernel {
 public:
  explicit PointsToOctreeOp(OpKernelConstruction* context) :
    OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("full_depth", &full_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("node_dis", &node_dis_));
    OP_REQUIRES_OK(context, context->GetAttr("node_feature", &node_feature_));
    OP_REQUIRES_OK(context, context->GetAttr("split_label", &split_label_));
    OP_REQUIRES_OK(context, context->GetAttr("adaptive", &adaptive_));
    OP_REQUIRES_OK(context, context->GetAttr("adp_depth", &adp_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("th_distance", &th_distance_));
    OP_REQUIRES_OK(context, context->GetAttr("th_normal", &th_normal_));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolate", &extrapolate_));
    OP_REQUIRES_OK(context, context->GetAttr("save_pts", &save_pts_));
    OP_REQUIRES_OK(context, context->GetAttr("key2xyz", &key2xyz_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data_in = context->input(0);
    CHECK_EQ(data_in.NumElements(), 1);

    // init the points
    Points point_cloud_;
    point_cloud_.set(data_in.flat<string>()(0).data());

    // check the points
    string msg;
    bool succ = point_cloud_.info().check_format(msg);
    CHECK(succ) << msg;

    // init the octree info
    OctreeInfo octree_info_;
    octree_info_.initialize(depth_, full_depth_, node_dis_,
      node_feature_, split_label_, adaptive_, adp_depth_,
      th_distance_, th_normal_, key2xyz_, extrapolate_,
      save_pts_, point_cloud_);

    // build the octree
    Octree octree_;
    octree_.build(octree_info_, point_cloud_);
    const vector<char>& octree_buf = octree_.buffer();

    // output
    Tensor* out_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, data_in.shape(), &out_data));
    string& out_str = out_data->flat<string>()(0);
    out_str.assign(octree_buf.begin(), octree_buf.end());
  }

 private:
  int depth_;
  int full_depth_;
  bool node_dis_;
  bool node_feature_;
  bool split_label_;
  bool adaptive_;
  int adp_depth_;
  float th_distance_;
  float th_normal_;
  bool extrapolate_;
  bool save_pts_;
  bool key2xyz_;
};


REGISTER_KERNEL_BUILDER(Name("PointsToOctree").Device(DEVICE_CPU), PointsToOctreeOp);
}  // namespace tensorflow
