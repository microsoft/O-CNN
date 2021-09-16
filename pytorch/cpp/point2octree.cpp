#include <octree/octree.h>

#include "ocnn.h"

Tensor points2octree(Tensor points, int depth, int full_depth, bool node_dis,
                     bool node_feature, bool split_label, bool adaptive,
                     int adp_depth, float th_normal, float th_distance,
                     bool extrapolate, bool save_pts, bool key2xyz) {
  // init the points
  Points point_cloud;
  point_cloud.set(points.data_ptr<uint8_t>());
  
  // check the points
  string msg;
  bool succ = point_cloud.info().check_format(msg);
  CHECK(succ) << msg;

  // init the octree info
  OctreeInfo octree_info;
  octree_info.initialize(depth, full_depth, node_dis, node_feature, split_label,
                         adaptive, adp_depth, th_distance, th_normal, key2xyz,
                         extrapolate, save_pts, point_cloud);

  // build the octree
  Octree octree_;
  octree_.build(octree_info, point_cloud);
  const vector<char>& octree_buf = octree_.buffer();

  // output
  size_t sz = octree_buf.size();
  Tensor output = torch::zeros({(int64_t)sz}, points.options());
  memcpy(output.data_ptr<uint8_t>(), octree_buf.data(), sz);
  return output;
}
