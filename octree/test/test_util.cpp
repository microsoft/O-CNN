#include <gtest/gtest.h>
#include "math_functions.h"
#include <cmath>
#include <mesh.h>
#include <filenames.h>

TEST(UtilTest, TestExtractPath) {
  EXPECT_EQ(extract_path("C:\\test\\test.txt"), "C:/test");
  EXPECT_EQ(extract_path("C:/test\\test.txt"), "C:/test");
  EXPECT_EQ(extract_path("C:/test"), "C:");
  EXPECT_EQ(extract_path("test.txt"), ".");
  EXPECT_EQ(extract_path("./test.txt"), ".");
}

TEST(UtilTest, TestExtractFilename) {
  EXPECT_EQ(extract_filename("C:\\test\\test.txt"), "test");
  EXPECT_EQ(extract_filename("C:/test\\test.txt"), "test");
  EXPECT_EQ(extract_filename("C:/test"), "test");
  EXPECT_EQ(extract_filename("test.txt"), "test");
  EXPECT_EQ(extract_filename("./test.txt"), "test");
  EXPECT_EQ(extract_filename("test"), "test");
}

TEST(UtilTest, TestExtractSuffix) {
  EXPECT_EQ(extract_suffix("C:\\test\\test.txt"), "txt");
  EXPECT_EQ(extract_suffix("C:/test\\test.TXT"), "txt");
  EXPECT_EQ(extract_suffix("C:/test"), "");
}

TEST(MeshTest, TestFaceCenter) {
  vector<float> V{ 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0};
  vector<int> F{0, 1, 3, 1, 2, 3};
  vector<float> center{ 1.0f / 3.0f, 1.0f / 3.0f, 0, 2.0f / 3.0f, 2.0f / 3.0f, 1.0f / 3.0f };
  vector<float> center_test;
  compute_face_center(center_test, V, F);
  ASSERT_EQ(center_test.size(), center.size());
  for (int i = 0; i < center.size(); ++i) {
    EXPECT_FLOAT_EQ(center[i], center_test[i]);
  }
}

TEST(MeshTest, TestFaceNormal) {
  vector<float> V{ 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0 };
  vector<int> F{ 0, 1, 3, 1, 2, 3 };
  float t = sqrtf(1.0f / 3.0f), q = sqrtf(3.0f) * 0.5f;
  vector<float> face_normal{0, 0, 1.0f, -t, -t, t };
  vector<float> face_area{ 0.5f, q };
  vector<float> normal_test, area_test;
  compute_face_normal(normal_test, area_test, V, F);
  ASSERT_EQ(normal_test.size(), face_normal.size());
  for (int i = 0; i < normal_test.size(); ++i) {
    EXPECT_FLOAT_EQ(normal_test[i], face_normal[i]);
  }

  ASSERT_EQ(area_test.size(), face_area.size());
  for (int i = 0; i < area_test.size(); ++i) {
    EXPECT_FLOAT_EQ(area_test[i], face_area[i]);
  }
}

TEST(MathTest, TestRotMatrix1) {
  float angles[3] = { 0.1f, 0.2f, 0.3f }, rot[9];
  rotation_matrix(rot, angles);

  float rotx[9], roty[9], rotz[9], rot_gt[9], tmp[9];
  float x[3] = { 1.0f, 0, 0 }, y[3] = { 0, 1.0f, 0 }, z[3] = { 0, 0, 1.0f };
  rotation_matrix(rotx, angles[0], x);
  rotation_matrix(roty, angles[1], y);
  rotation_matrix(rotz, angles[2], z);
  matrix_prod(tmp, rotx, roty, 3, 3, 3);
  matrix_prod(rot_gt, tmp, rotz, 3, 3, 3);

  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(rot[i], rot_gt[i]);
  }
}

TEST(MathTest, TestRotMatrix2) {
  const float kPI = 3.1415925f;
  float angle1[3] = { -10.0, -20.0f, -210.0f }, rot1[9];
  float angle2[3] = { 350.0f, 340.0f, 150.0f }, rot2[9];
  for (int i = 0; i < 3; ++i) {
    angle1[i] = angle1[i] * kPI / 180.0f;
    angle2[i] = angle2[i] * kPI / 180.0f;
  }
  rotation_matrix(rot1, angle1);
  rotation_matrix(rot2, angle2);

   for (int i = 0; i < 9; ++i) {
     ASSERT_NEAR(rot1[i], rot2[i], 1.0e-5);
  }
}


//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//  return RUN_ALL_TESTS();
//}