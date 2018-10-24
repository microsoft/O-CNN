#include "caffe/test/test_octree.hpp"
#include "caffe/layers/octree_property_layer.hpp"
#include "caffe/layers/octree_tile_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctreePropertyLayerTest : public OctreeTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctreePropertyLayerTest() {}
  virtual void SetUp() {}
};


TYPED_TEST_CASE(OctreePropertyLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctreePropertyLayerTest, TestForwardFunc) {
  typedef typename TypeParam::Dtype Dtype;

  // prepare testing data
  string octree1("octree_3"), octree2("octree_4");
  this->load_test_data(vector<string> {octree1, octree2});
  this->set_octree_batch();
  size_t sz1 = 0, sz2 = 0;
  const char* oct_ptr1 = get_test_octree(octree1.c_str(), &sz1);
  const char* oct_ptr2 = get_test_octree(octree2.c_str(), &sz2);
  OctreeParser parser1, parser2;
  parser1.set_cpu(oct_ptr1);
  parser2.set_cpu(oct_ptr2);

  const OctreeInfo& oct_info = this->octree_parser_.info();
  const int depth = oct_info.depth();
  for (int d = 0; d < depth; ++d) {
    // forward
    const int curr_depth = d;
    const int prop_num = 3, signal_channel = 1;
    const string content_flag = "key,child,split";
    LayerParameter layer_param;
    layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
    layer_param.mutable_octree_param()->set_content_flag(content_flag);
    layer_param.mutable_octree_param()->set_signal_channel(signal_channel);

    OctreePropertyLayer<Dtype> layer(layer_param);
    vector<Blob<Dtype>*> blob_bottom_vec, blob_top_vec;
    for (int i = 0; i < prop_num; ++i) {
      blob_top_vec.push_back(new Blob<Dtype>());
    }
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer.Forward(blob_bottom_vec, blob_top_vec);

    int nnum1 = parser1.info().node_num(d);
    int nnum2 = parser2.info().node_num(d);
    int nnumb = nnum1 + nnum2;

    // compare key
    const unsigned int* key1 = parser1.key_cpu(d);
    const unsigned int* key2 = parser2.key_cpu(d);
    const int* key_batch = reinterpret_cast<const int*>(blob_top_vec[0]->cpu_data());
    int height = blob_top_vec[0]->shape(2);
    int height_target = sizeof(Dtype) == 8 ? (nnumb + 1) / 2 : nnumb;
    ASSERT_EQ(height, height_target);
    for (int i = 0; i < nnumb; ++i) {
      int src = 0, idx = 0;
      if (i < nnum1) {
        src = key1[i];
      } else {
        src = key2[i - nnum1];
        idx = 1;
      }
      unsigned char* ptr = reinterpret_cast<unsigned char*>(&src);
      ptr[3] = idx; // !!! todo: deal with octree depth > 8
      int des = key_batch[i];
      ASSERT_EQ(des, src) << "Key : " << nnum1 << ", " << nnum2 << ", " << i;
    }


    // check children
    const int* child1 = parser1.children_cpu(d);
    const int* child2 = parser2.children_cpu(d);
    const int* child_batch = reinterpret_cast<const int*>(blob_top_vec[1]->cpu_data());
    height = blob_top_vec[1]->shape(2);
    height_target = sizeof(Dtype) == 8 ? (nnumb + 1) / 2 : nnumb;
    ASSERT_EQ(height, height_target);
    for (int i = 0; i < nnumb; ++i) {
      int src = 0;
      if (i < nnum1) {
        src = child1[i];
      } else {
        src = child2[i - nnum1];
        if (src != -1) src += parser1.info().node_num_nempty(d);
      }
      int des = child_batch[i];
      ASSERT_EQ(des, src) << "Children : " << nnum1 << ", " << nnum2 << ", " << i;
    }

    // split label
    const float* split1 = parser1.split_cpu(d);
    const float* split2 = parser2.split_cpu(d);
    const Dtype* split_batch = blob_top_vec[2]->cpu_data();
    height = blob_top_vec[2]->shape(2);
    ASSERT_EQ(height, nnumb);
    for (int i = 0; i < nnumb; ++i) {
      float src = i < nnum1 ? split1[i] : split2[i - nnum1];
      Dtype des = split_batch[i];
      ASSERT_EQ(des, src) << "Split : " << nnum1 << ", " << nnum2 << ", " << i;
    }

    // release
    for (int i = 0; i < prop_num; ++i) {
      delete blob_top_vec[i];
    }
  }
}

TYPED_TEST(OctreePropertyLayerTest, TestForwardFuncData) {
  typedef typename TypeParam::Dtype Dtype;

  // prepare testing data
  string octree1("octree_3"), octree2("octree_4");
  this->load_test_data(vector<string> {octree1, octree2});
  this->set_octree_batch();
  size_t sz1 = 0, sz2 = 0;
  const char* oct_ptr1 = get_test_octree(octree1.c_str(), &sz1);
  const char* oct_ptr2 = get_test_octree(octree2.c_str(), &sz2);
  OctreeParser parser1, parser2;
  parser1.set_cpu(oct_ptr1);
  parser2.set_cpu(oct_ptr2);

  const OctreeInfo& oct_info = this->octree_parser_.info();
  const int depth = oct_info.depth();
  for (int d = 0; d < depth; ++d) {
    // forward
    const int curr_depth = d;
    const int prop_num = 1, signal_channel = 4;
    const string content_flag = "feature";
    LayerParameter layer_param;
    layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
    layer_param.mutable_octree_param()->set_content_flag(content_flag);
    layer_param.mutable_octree_param()->set_signal_channel(signal_channel);

    OctreePropertyLayer<Dtype> layer(layer_param);
    vector<Blob<Dtype>*> blob_bottom_vec, blob_top_vec;
    for (int i = 0; i < prop_num; ++i) {
      blob_top_vec.push_back(new Blob<Dtype>());
    }
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer.Forward(blob_bottom_vec, blob_top_vec);

    int nnum1 = parser1.info().node_num(d);
    int nnum2 = parser2.info().node_num(d);
    int nnumb = nnum1 + nnum2;

    // feature
    const float* feature1 = parser1.feature_cpu(d);
    const float* feature2 = parser2.feature_cpu(d);
    const Dtype* feature_batch = blob_top_vec[0]->cpu_data();
    ASSERT_EQ(blob_top_vec[0]->shape(2), nnumb);
    ASSERT_EQ(blob_top_vec[0]->shape(1), signal_channel);
    for (int c = 0; c < signal_channel; ++c) {
      for (int i = 0; i < nnumb; ++i) {
        float src = i < nnum1 ? feature1[c * nnum1 +  i] : feature2[c * nnum2 + i - nnum1];
        Dtype des = feature_batch[c * nnumb +  i];
        ASSERT_EQ(des, src) << "Feature : " << nnum1 << ", " << nnum2 << ", " << i;
      }
    }

    // release
    for (int i = 0; i < prop_num; ++i) {
      delete blob_top_vec[i];
    }
  }
}

}  // namespace caffe