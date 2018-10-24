#include "caffe/test/test_octree.hpp"
#include "caffe/layers/octree_property_layer.hpp"
#include "caffe/layers/octree_grow_layer.hpp"
#include "caffe/layers/octree_tile_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctreeGrowLayerTest : public OctreeTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctreeGrowLayerTest() {}
  virtual void SetUp() {}
};


TYPED_TEST_CASE(OctreeGrowLayerTest, TestDtypesAndDevices);


TYPED_TEST(OctreeGrowLayerTest, TestFullOctree) {
  typedef typename TypeParam::Dtype Dtype;

  // prepare testing data
  string octree1("octree_3");
  size_t sz1 = 0;
  const char* oct_ptr1 = get_test_octree(octree1.c_str(), &sz1);
  OctreeParser parser1;
  parser1.set_cpu(oct_ptr1);

  // generate full octree
  const int full_octree = true;
  const int curr_depth = 2, batch_size = 1;

  LayerParameter layer_param;
  layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
  layer_param.mutable_octree_param()->set_batch_size(batch_size);
  layer_param.mutable_octree_param()->set_full_octree(full_octree);

  OctreeGrowLayer<Dtype> layer(layer_param);
  vector<Blob<Dtype>*> blob_bottom_vec, blob_top_vec;
  blob_top_vec.push_back(new Blob<Dtype>());
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  layer.Forward(blob_bottom_vec, blob_top_vec);

  // test octree info.
  OctreeParser parser;
  parser.set_cpu(blob_top_vec[0]->cpu_data());
  ASSERT_EQ(parser.info().batch_size(), batch_size);
  ASSERT_EQ(parser.info().depth(), curr_depth);
  ASSERT_EQ(parser.info().full_layer(), curr_depth);
  ASSERT_EQ(parser.info().is_adaptive(), false);
  ASSERT_EQ(parser.info().key2xyz(), true);

  // test key
  const unsigned int* key1 = parser1.key_cpu(curr_depth);
  const unsigned int* key_batch = parser.key_cpu(curr_depth);
  int nnum1 = parser1.info().node_num(curr_depth);
  int nnum_batch = parser.info().node_num(curr_depth);
  ASSERT_EQ(nnum1, nnum_batch);
  for (int i = 0; i < nnum1; ++i) {
    const char* ptr1 = reinterpret_cast<const char*>(key1 + i);
    const char* ptr_batch = reinterpret_cast<const char*>(key_batch + i);
    for (int c = 0; c < 3; ++c) {
      ASSERT_EQ(ptr1[c], ptr_batch[c]) << "Key : " << i;
    }
  }

  // check children
  const int* child_batch = parser.children_cpu(curr_depth);
  for (int i = 0; i < nnum1; ++i) {
    ASSERT_EQ(child_batch[i], i) << "Children : " << i;
  }

  // release
  delete blob_top_vec[0];
}

}  // namespace caffe