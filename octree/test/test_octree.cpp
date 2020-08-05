#include <gtest/gtest.h>
#include <points.h>
#include <octree.h>
#include <octree_value.h>
#include "math_functions.h"

class OctreeTest : public ::testing::Test {
 protected:
  void gen_test_point() {
    vector<float> pt{1.0f, 1.0f, 0.0f};
    vector<float> normal{ 1.0f, 0.0f, 0.0f};
    vector<float> feature{ 1.0f, -1.0f, 2.0f};
    vector<float> label{ 0.0f};
    points.set_points(pt, normal, feature, label);
  }

  void gen_test_pointcloud() {
    vector<float> pts { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0625f, 1.0625f, 0.0f};
    vector<float> normals { 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
    vector<float> features{ 1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f };
    vector<float> labels { 0.0f, 2.0f, 2.0f };
    points.set_points(pts, normals, features, labels);
  }

  void build_octree() {
    octree_.build(oct_info_, points);
  }

  void trim_octree() {
    octree_.trim_octree();
  }

  void set_octree_info(const bool adaptive, const bool key2xyz, const bool split_label,
      const float* bbmin, const float* bbmax) {
    const bool node_dis = true, node_feature = false;
    const int depth = 5, full_layer = 1, adaptive_layer = 3;
    // The normal threshold is very large and has no effect to the adaptive octree
    const float th_normal = 5.0f, th_distance = 3.0f;

    oct_info_.reset();
    oct_info_.set_batch_size(1);
    oct_info_.set_depth(depth);
    oct_info_.set_full_layer(full_layer);
    oct_info_.set_adaptive_layer(adaptive_layer);
    oct_info_.set_adaptive(adaptive);
    oct_info_.set_node_dis(node_dis);
    oct_info_.set_key2xyz(key2xyz);
    oct_info_.set_threshold_normal(th_normal);
    oct_info_.set_threshold_dist(th_distance);
    oct_info_.set_bbox(bbmin, bbmax);

    // by default, the octree contains Key and Child
    int channel = (key2xyz && depth > 8) ? 2 : 1;
    oct_info_.set_channel(OctreeInfo::kKey, channel);
    oct_info_.set_location(OctreeInfo::kKey, -1);
    oct_info_.set_channel(OctreeInfo::kChild, 1);
    oct_info_.set_location(OctreeInfo::kChild, -1);

    // set feature
    const PointsInfo& pt_info = points.info();
    channel = pt_info.channel(PointsInfo::kNormal) + pt_info.channel(PointsInfo::kFeature);
    if (node_dis) channel += 1;
    oct_info_.set_channel(OctreeInfo::kFeature, channel);
    // location = -1 means the features exist on every node
    int location = (node_feature || adaptive) ? -1 : depth;
    oct_info_.set_location(OctreeInfo::kFeature, location);

    // set label
    if (pt_info.channel(PointsInfo::kLabel) == 1) {
      oct_info_.set_channel(OctreeInfo::kLabel, 1);
      location = (node_feature || adaptive) ? -1 : depth;
      oct_info_.set_location(OctreeInfo::kLabel, location);
    }

    // set split label
    if (split_label) {
      oct_info_.set_channel(OctreeInfo::kSplit, 1);
      oct_info_.set_location(OctreeInfo::kSplit, -1);
    }

    // Skip nnum_[], nnum_cum_[], nnum_nempty_[] and ptr_dis_[],
    // these three properties can only be set when the octree is built.
  }

 protected:
  Points points;
  Octree octree_;
  OctreeInfo oct_info_;

};

TEST_F(OctreeTest, TestOctreeBuild) {
  const float bbmin[] = { 0.0f, 0.0f, 0.0f };
  const float bbmax[] = { 2.0f, 2.0f, 2.0f };
  const bool adaptive = false, key2xyz = false, calc_split_label = true;
  this->gen_test_pointcloud();
  this->set_octree_info(adaptive, key2xyz, calc_split_label, bbmin, bbmax);
  this->build_octree();

  const OctreeInfo& info = octree_.info();
  const int depth = 5, full_layer = 1;
  EXPECT_EQ(info.depth(), depth);
  EXPECT_EQ(info.full_layer(), full_layer);

  // test node number
  const int nnum[] = { 1, 8, 16, 16, 16, 16 };
  const int nnum_cum[] = { 0, 1, 9, 25, 41, 57, 73 };
  const int nnum_nempty[] = { 1, 2, 2, 2, 2, 3 };
  for (int d = 0; d <= depth; ++d) {
    EXPECT_EQ(info.node_num(d), nnum[d]);
    EXPECT_EQ(info.node_num_cum(d), nnum_cum[d]);
    EXPECT_EQ(info.node_num_nempty(d), nnum_nempty[d]);
  }
  EXPECT_EQ(info.node_num_cum(depth + 1), nnum_cum[depth + 1]);

  // test the key
  const uintk keys[] = {
    0, 0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7, 48, 49, 50, 51, 52, 53, 54, 55,
    0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 386, 387, 388, 389, 390, 391,
    0, 1, 2, 3, 4, 5, 6, 7, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079,
    0, 1, 2, 3, 4, 5, 6, 7, 24576, 24577, 24578, 24579, 24580, 24581, 24582, 24583
  };
  for (int d = 0, j = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const uintk* key_d = octree_.key_cpu(d);
    for (int i = 0; i < nnum_d; ++i, j++) {
      EXPECT_EQ(key_d[i], keys[j]);
    }
  }

  // test the children
  const int children[] = {
    0, 0, -1, -1, -1, -1, -1, 1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 2, -1,
  };
  for (int d = 0, j = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const int* child_d = octree_.children_cpu(d);
    for (int i = 0; i < nnum_d; ++i, j++) {
      EXPECT_EQ(child_d[i], children[j]);
    }
  }

  // test the signal
  const float features[] = {
    1.0f, 0, 0, 0, 0, 0, 0, 0, -1.0f, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // normals
    -0.57735f, 0, 0, 0, 0, 0, 0, 0, 0.57735f, 0, 0, 0, 0, 0, -0.57735f, 0, // displacement
    1.0f, 0, 0, 0, 0, 0, 0, 0, 2.0f, 0, 0, 0, 0, 0, 3.0f, 0,
    -1.0f, 0, 0, 0, 0, 0, 0, 0, -2.0f, 0, 0, 0, 0, 0, -3.0f, 0  // other features
  };
  const int channel = 6;
  const int nnum_d = info.node_num(depth);
  const float* feature_d = octree_.feature_cpu(depth);
  EXPECT_EQ(channel, info.channel(OctreeInfo::kFeature));
  for (int i = 0; i < nnum_d * channel; ++i) {
    EXPECT_EQ(feature_d[i], features[i]);
  }

  // test the label
  const float labels[] = {
    0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    2.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 2.0f, -1.0f
  };
  const float* label_d = octree_.label_cpu(depth);
  for (int i = 0; i < nnum_d; ++i) {
    EXPECT_EQ(label_d[i], labels[i]);
  }

  // test the split label
  const float split_labels[] = {
    1.0f, 1.0f, 0, 0, 0, 0, 0, 1.0f, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0, 0, 0, 0, 0, 1.0f, 0,
  };
  for (int d = 0, j = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const float* split_d = octree_.split_cpu(d);
    for (int i = 0; i < nnum_d; ++i, j++) {
      EXPECT_EQ(split_d[i], split_labels[j]);
    }
  }

  // test the fval
  OctreeValue octree_value(&octree_);
  EXPECT_EQ(octree_value.fval(16.5f, 16.5f, 0.5f).first, -0.5f);
}

TEST_F(OctreeTest, TestOctreeTrim) {
  const float bbmin[] = { 0.0f, 0.0f, 0.0f };
  const float bbmax[] = { 2.0f, 2.0f, 2.0f };
  const bool adaptive = true, key2xyz = false, calc_split_label = true;
  this->gen_test_pointcloud();
  this->set_octree_info(adaptive, key2xyz, calc_split_label, bbmin, bbmax);
  this->build_octree();
  this->trim_octree();

  const OctreeInfo& info = octree_.info();
  const int depth = 5, full_layer = 1, depth_adpt = 3;
  EXPECT_EQ(info.depth(), depth);
  EXPECT_EQ(info.full_layer(), full_layer);
  EXPECT_EQ(info.adaptive_layer(), depth_adpt);

  // test node number
  const int nnum[] = { 1, 8, 16, 16, 16, 8 };
  const int nnum_cum[] = { 0, 1, 9, 25, 41, 57, 65 };
  const int nnum_nempty[] = { 1, 2, 2, 2, 1, 1 };
  for (int d = 0; d <= depth; ++d) {
    EXPECT_EQ(info.node_num(d), nnum[d]);
    EXPECT_EQ(info.node_num_cum(d), nnum_cum[d]);
    EXPECT_EQ(info.node_num_nempty(d), nnum_nempty[d]);
  }
  EXPECT_EQ(info.node_num_cum(depth + 1), nnum_cum[depth + 1]);

  // test the key
  const uintk keys[] = {
    0, 0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7, 48, 49, 50, 51, 52, 53, 54, 55,
    0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 386, 387, 388, 389, 390, 391,
    0, 1, 2, 3, 4, 5, 6, 7, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079,
    0, 1, 2, 3, 4, 5, 6, 7
  };
  for (int d = 0, j = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const uintk* key_d = octree_.key_cpu(d);
    for (int i = 0; i < nnum_d; ++i, j++) {
      EXPECT_EQ(key_d[i], keys[j]);
    }
  }

  // test the children
  const int children[] = {
    0, 0, -1, -1, -1, -1, -1, 1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1, -1, -1/**/, -1, -1, -1, -1, -1, -1, -1,
    0, -1, -1, -1, -1, -1, -1, -1
  };
  for (int d = 0, j = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const int* child_d = octree_.children_cpu(d);
    for (int i = 0; i < nnum_d; ++i, j++) {
      EXPECT_EQ(child_d[i], children[j]);
    }
  }

  //// test the signal
  const float feature0[] = {
    0, 1.0f, 0, -0.180422f, 2.0f, -2.0f
  };
  const float feature1[] = {
    1.0f, 0, 0, 0, 0, 0, -0.707107f, 0,
    0, 0, 0, 0, 0, 0, 0.707107f, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    -0.57735f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 2.5f, 0,
    -1.0f, 0, 0, 0, 0, 0, -2.5f, 0
  };
  const float feature2[] = {
    1.0f, 0, 0, 0, 0, 0, 0, 0, -0.707107f, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0.707107f, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.57735f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 2.5f, 0, 0, 0, 0, 0, 0, 0,
    -1.0f, 0, 0, 0, 0, 0, 0, 0, -2.5f, 0, 0, 0, 0, 0, 0, 0
  };
  const float* feature3 = feature2;
  const float* feature4 = feature2;
  const float feature5[] = {
    1.0f, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    -0.57735f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0,
    -1.0f, 0, 0, 0, 0, 0, 0, 0
  };
  const float* features[] = { feature0, feature1, feature2, feature3, feature4, feature5};
  const int channel = 6;
  const int nnum_d = info.node_num(depth);
  EXPECT_EQ(channel, info.channel(OctreeInfo::kFeature));

  for (int d = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const float* feature_d = octree_.feature_cpu(d);
    for (int i = 0; i < nnum_d * channel; ++i) {
      EXPECT_FLOAT_EQ(features[d][i], feature_d[i]) << i;
    }
  }

  // test the label
  const float labels[] = {
    2.0f, 0, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 2.0f, -1.0f,
    0, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    2.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    0, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    2.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    0, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    2.0f/**/, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    0, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
  };
  for (int d = 0, j = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const float* label_d = octree_.label_cpu(d);
    for (int i = 0; i < nnum_d; ++i, j++) {
      EXPECT_EQ(label_d[i], labels[j]);
    }
  }

  // test the split label
  const float split_labels[] = {
    1.0f, 1.0f, 0, 0, 0, 0, 0, 1.0f, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0, 2.0f, 0, 0, 0, 0, 0, 0, 0,
    1.0f, 0, 0, 0, 0, 0, 0, 0
  };
  for (int d = 0, j = 0; d <= depth; ++d) {
    const int nnum_d = info.node_num(d);
    const float* split_d = octree_.split_cpu(d);
    for (int i = 0; i < nnum_d; ++i, j++) {
      EXPECT_EQ(split_d[i], split_labels[j]);
    }
  }
}