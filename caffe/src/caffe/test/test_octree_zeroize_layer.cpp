
#include "caffe/test/test_octree.hpp"
#include "caffe/layers/octree_zeroize_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctreeZeroizeLayerTest : public OctreeTest<TypeParam> {
};

TYPED_TEST_CASE(OctreeZeroizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctreeZeroizeLayerTest, TestForwardBackwardFunc) {
  // forward test data
  typedef typename TypeParam::Dtype Dtype;
  vector<Dtype> btm0{ 1, 2, 3, 4, 5, 6 };
  vector<Dtype> top0{ 1, 0, 3, 4, 0, 6 };
  vector<Dtype> btm1{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  vector<Dtype> top1{ 1, 0, 3, 4, 0, 6, 7, 0, 9 };
  vector<Dtype> mask{ 0, 1, 2 };
  Blob<Dtype> btm_blob[3], top_blob[2];
  btm_blob[0].Reshape(vector<int> {1, 2, 3, 1});
  btm_blob[1].Reshape(vector<int> {1, 3, 3, 1});
  btm_blob[2].Reshape(vector<int> {1, 1, 3, 1});
  std::copy(btm0.begin(), btm0.end(), btm_blob[0].mutable_cpu_data());
  std::copy(btm1.begin(), btm1.end(), btm_blob[1].mutable_cpu_data());
  std::copy(mask.begin(), mask.end(), btm_blob[2].mutable_cpu_data());
  vector<Blob<Dtype>*> vec_btm_blob{ btm_blob, btm_blob + 1, btm_blob + 2 };
  vector<Blob<Dtype>*> vec_top_blob{ top_blob, top_blob + 1 };

  // forward
  LayerParameter layer_param;
  layer_param.mutable_octree_param()->add_mask(1);
  OctreeZeroizeLayer<Dtype> layer(layer_param);
  layer.SetUp(vec_btm_blob, vec_top_blob);
  layer.Forward(vec_btm_blob, vec_top_blob);

  // check
  ASSERT_TRUE(top_blob[0].shape() == btm_blob[0].shape());
  const Dtype* top_data0 = top_blob[0].cpu_data();
  for (int i = 0; i < top0.size(); ++i) {
    ASSERT_EQ(top_data0[i], top0[i]);
  }
  ASSERT_TRUE(top_blob[1].shape() == btm_blob[1].shape());
  const Dtype* top_data1 = top_blob[1].cpu_data();
  for (int i = 0; i < top1.size(); ++i) {
    ASSERT_EQ(top_data1[i], top1[i]);
  }

  // backward test data
  std::copy(btm0.begin(), btm0.end(), top_blob[0].mutable_cpu_diff());
  std::copy(btm1.begin(), btm1.end(), top_blob[1].mutable_cpu_diff());
  vector<bool> propagate_down{ true, true, true };

  // backward
  layer.Backward(vec_top_blob, propagate_down, vec_btm_blob);

  // check
  const Dtype* bottom_diff0 = btm_blob[0].cpu_diff();
  for (int i = 0; i < top0.size(); ++i) {
    ASSERT_EQ(bottom_diff0[i], top0[i]);
  }
  const Dtype* bottom_diff1 = btm_blob[1].cpu_diff();
  for (int i = 0; i < top1.size(); ++i) {
    ASSERT_EQ(bottom_diff1[i], top1[i]) << i;
  }
}

}  // namespace caffe