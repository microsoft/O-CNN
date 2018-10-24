
#include "caffe/test/test_octree.hpp"
#include "caffe/layers/octree_mask_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctreeMaskLayerTest : public OctreeTest<TypeParam> {
};

TYPED_TEST_CASE(OctreeMaskLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctreeMaskLayerTest, TestForwardBackwardFunc) {
  // forward test data
  typedef typename TypeParam::Dtype Dtype;
  vector<Dtype> btm0 { 1, 2, 3, 4, 5, 6 };
  vector<Dtype> btm1 { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  vector<int> mask { 0, 1, 2 };
  Blob<Dtype> btm_blob[3], top_blob[2];
  btm_blob[0].Reshape(vector<int> {1, 2, 3, 1});
  btm_blob[1].Reshape(vector<int> {1, 3, 3, 1});
  btm_blob[2].Reshape(vector<int> {1, 1, 3, 1});
  std::copy(btm0.begin(), btm0.end(), btm_blob[0].mutable_cpu_data());
  std::copy(btm1.begin(), btm1.end(), btm_blob[1].mutable_cpu_data());
  std::copy(mask.begin(), mask.end(), btm_blob[2].mutable_cpu_data());
  vector<Blob<Dtype>*> vec_btm_blob{ btm_blob, btm_blob + 1, btm_blob + 2 };
  vector<Blob<Dtype>*> vec_top_blob{ top_blob, top_blob + 1};

  // forward
  LayerParameter layer_param;
  layer_param.mutable_octree_param()->add_mask(2);
  OctreeMaskLayer<Dtype> layer(layer_param);
  layer.SetUp(vec_btm_blob, vec_top_blob);
  layer.Forward(vec_btm_blob, vec_top_blob);

  // check
  vector<int> top_shape = { 1, 2, 1, 1 };
  ASSERT_TRUE(top_blob[0].shape() == top_shape);
  const Dtype* top_data = top_blob[0].cpu_data();
  ASSERT_EQ(top_data[0], 3.0);
  ASSERT_EQ(top_data[1], 6.0);

  top_shape = { 1, 3, 1, 1 };
  ASSERT_TRUE(top_blob[1].shape() == top_shape);
  top_data = top_blob[1].cpu_data();
  ASSERT_EQ(top_data[0], 3.0);
  ASSERT_EQ(top_data[1], 6.0);
  ASSERT_EQ(top_data[2], 9.0);

  // backward test data
  vector<Dtype> top0{ 1, 2 };
  vector<Dtype> top1{ 1, 2, 3 };
  std::copy(top0.begin(), top0.end(), top_blob[0].mutable_cpu_diff());
  std::copy(top1.begin(), top1.end(), top_blob[1].mutable_cpu_diff());
  vector<bool> propagate_down{ true, true, true };

  // backward
  layer.Backward(vec_top_blob, propagate_down, vec_btm_blob);

  // check
  vector<Dtype> btm_diff0{ 0, 0, 1, 0, 0, 2 };
  vector<Dtype> btm_diff1{ 0, 0, 1, 0, 0, 2, 0, 0, 3 };
  const Dtype* btm_diff = btm_blob[0].cpu_diff();
  for (int i = 0; i < btm_diff0.size(); ++i) {
    ASSERT_EQ(btm_diff[i], btm_diff0[i]);
  }
  btm_diff = btm_blob[1].cpu_diff();
  for (int i = 0; i < btm_diff1.size(); ++i) {
    ASSERT_EQ(btm_diff[i], btm_diff1[i]);
  }
}

}  // namespace caffe 