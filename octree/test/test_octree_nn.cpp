#include <gtest/gtest.h>
#include <octree_nn.h>


TEST(VecResizeTest, TestVecResize) {
  vector<int> vec0{ 3 }, gt0{ 3, 3, 3 };
  resize_with_last_val(vec0, 3);
  ASSERT_EQ(vec0, gt0);

  vector<int> vec1{ 3, 1, 1, 3 }, gt1{3, 1, 1};
  resize_with_last_val(vec1, 3);
  ASSERT_EQ(vec1, gt1);

  vector<int> vec2, gt2;
  resize_with_last_val(vec2, 3);
  ASSERT_EQ(vec2, gt2);
}

