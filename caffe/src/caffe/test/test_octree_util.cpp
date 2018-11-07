#include "caffe/util/octree.hpp"

#include "caffe/test/test_octree.hpp"

namespace caffe {

template <typename TypeParam>
class OctreeUtilTest : public OctreeTest<TypeParam> {
 protected:
  typedef typename TypeParam::Dtype Dtype;
  OctreeUtilTest()  {}
  virtual void SetUp() {}

  // todo: add test for the label
  void test_octree_batch(const string& octree1, const string& octree2) {
    size_t sz1 = 0, sz2 = 0;
    const char* oct_ptr1 = get_test_octree(octree1.c_str(), &sz1);
    const char* oct_ptr2 = get_test_octree(octree2.c_str(), &sz2);

    this->load_test_data(vector<string> {octree1, octree2});
    Blob<Dtype> octree_batch;
    octree::merge_octrees(octree_batch, this->octree_buffer_);

    OctreeParser parser_batch, parser1, parser2;
    parser_batch.set_cpu(octree_batch.cpu_data());
    parser1.set_cpu(oct_ptr1);
    parser2.set_cpu(oct_ptr2);

    int depth = parser_batch.info().depth();
    ASSERT_EQ(depth, parser1.info().depth());
    ASSERT_EQ(depth, parser2.info().depth());

    for (int d = 0; d <= depth; ++d) {
      int nnum1 = parser1.info().node_num(d);
      int nnum2 = parser2.info().node_num(d);
      int nnumb = parser_batch.info().node_num(d);
      ASSERT_EQ(nnum1 + nnum2, nnumb);

      // compare key
      ASSERT_TRUE(parser_batch.info().key2xyz()); // todo: compare the other condition
      ASSERT_TRUE(parser_batch.info().has_property(OctreeInfo::kKey));
      const unsigned int* key1 = parser1.key_cpu(d);
      const unsigned int* key2 = parser2.key_cpu(d);
      const unsigned int* key_batch = parser_batch.key_cpu(d);
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

      // compare children
      ASSERT_TRUE(parser_batch.info().has_property(OctreeInfo::kChild));
      const int* child1 = parser1.children_cpu(d);
      const int* child2 = parser2.children_cpu(d);
      const int* child_batch = parser_batch.children_cpu(d);
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

      // compare data
      ASSERT_TRUE(parser_batch.info().has_property(OctreeInfo::kFeature));
      const int channel = parser_batch.info().channel(OctreeInfo::kFeature);
      ASSERT_EQ(channel, parser1.info().channel(OctreeInfo::kFeature));
      ASSERT_EQ(channel, parser2.info().channel(OctreeInfo::kFeature));
      if (parser1.info().locations(OctreeInfo::kFeature) == depth) {
        if (d != depth) continue;

        const float* feature_batch = parser_batch.feature_cpu(d);
        const float* feature1 = parser1.feature_cpu(d);
        const float* feature2 = parser2.feature_cpu(d);
        for (int c = 0; c < channel; ++c) {
          for (int i = 0; i < nnumb; ++i) {
            float des = feature_batch[c * nnumb + i];
            float src = i < nnum1 ? feature1[c * nnum1 + i] : feature2[c * nnum2 + i - nnum1];
            ASSERT_EQ(des, src) << "Data : " << nnum1 << ", " << nnum2 << ", " << i;
          }
        }
      } else {
        const float* feature_batch = parser_batch.feature_cpu(d);
        const float* feature1 = parser1.feature_cpu(d);
        const float* feature2 = parser2.feature_cpu(d);
        for (int c = 0; c < channel; ++c) {
          for (int i = 0; i < nnumb; ++i) {
            int nnum_cum_d = parser_batch.info().node_num_cum(d);
            float des = feature_batch[c * nnumb + i];
            float src = i < nnum1 ? feature1[c * nnum1 + i] : feature2[c * nnum2 + i - nnum1];
            ASSERT_EQ(des, src) << "Data : " << nnum1 << ", " << nnum2 << ", " << i;
          }
        }
      }

      // comopare the split label
      if (!parser_batch.info().has_property(OctreeInfo::kSplit)) continue;
      const float* split1 = parser1.split_cpu(d);
      const float* split2 = parser2.split_cpu(d);
      const float* split_batch = parser_batch.split_cpu(d);
      for (int i = 0; i < nnumb; ++i) {
        float src = i < nnum1 ? split1[i] : split2[i - nnum1];
        float des = split_batch[i];
        ASSERT_EQ(des, src) << "Split : " << nnum1 << ", " << nnum2 << ", " << i;
      }
    }
  }

  void test_octree_batch_legacy(const string& octree1, const string& octree2) {
    this->load_test_data(vector<string> {octree1, octree2});
    this->set_octree_batch();

    size_t sz1 = 0, sz2 = 0;
    const char* oct_ptr1 = get_test_octree(octree1.c_str(), &sz1);
    const char* oct_ptr2 = get_test_octree(octree2.c_str(), &sz2);

    OctreeParser parser_batch, parser1, parser2;
    parser_batch.set_cpu(this->octree_batch_->octree_.cpu_data());
    parser1.set_cpu(oct_ptr1);
    parser2.set_cpu(oct_ptr2);

    int depth = parser_batch.info().depth();
    ASSERT_EQ(depth, parser1.info().depth());
    ASSERT_EQ(depth, parser2.info().depth());

    for (int d = 0; d <= depth; ++d) {
      int nnum1 = parser1.info().node_num(d);
      int nnum2 = parser2.info().node_num(d);
      int nnumb = parser_batch.info().node_num(d);
      ASSERT_EQ(nnum1 + nnum2, nnumb);

      // compare key
      ASSERT_TRUE(parser_batch.info().key2xyz()); // todo: compare the other condition
      ASSERT_TRUE(parser_batch.info().has_property(OctreeInfo::kKey));
      const int* key1 = parser1.key_cpu(d);
      const int* key2 = parser2.key_cpu(d);
      const int* key_batch = parser_batch.key_cpu(d);
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

      // compare children
      ASSERT_TRUE(parser_batch.info().has_property(OctreeInfo::kChild));
      const int* child1 = parser1.children_cpu(d);
      const int* child2 = parser2.children_cpu(d);
      const int* child_batch = parser_batch.children_cpu(d);
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

      // compare data
      const int channel = this->octree_batch_->data_.shape(1);
      ASSERT_EQ(channel, parser1.info().channel(OctreeInfo::kFeature));
      ASSERT_EQ(channel, parser2.info().channel(OctreeInfo::kFeature));
      if (parser1.info().locations(OctreeInfo::kFeature) == depth) {
        if (d != depth) continue;

        ASSERT_EQ(nnumb, this->octree_batch_->data_.shape(2));
        const Dtype* feature_batch = this->octree_batch_->data_.cpu_data();
        const float* feature1 = parser1.feature_cpu(d);
        const float* feature2 = parser2.feature_cpu(d);
        for (int c = 0; c < channel; ++c) {
          for (int i = 0; i < nnumb; ++i) {
            float des = feature_batch[c * nnumb + i];
            float src = i < nnum1 ? feature1[c * nnum1 + i] : feature2[c * nnum2 + i - nnum1];
            ASSERT_EQ(des, src) << "Data : " << nnum1 << ", " << nnum2 << ", " << i;
          }
        }
      } else {
        ASSERT_EQ(parser_batch.info().total_nnum(), this->octree_batch_->data_.shape(2));
        const Dtype* feature_batch = this->octree_batch_->data_.cpu_data();
        const float* feature1 = parser1.feature_cpu(d);
        const float* feature2 = parser2.feature_cpu(d);
        for (int c = 0; c < channel; ++c) {
          for (int i = 0; i < nnumb; ++i) {
            int nnum_cum_d = parser_batch.info().node_num_cum(d);
            float des = feature_batch[channel * nnum_cum_d + c * nnumb + i];
            float src = i < nnum1 ? feature1[c * nnum1 + i] : feature2[c * nnum2 + i - nnum1];
            ASSERT_EQ(des, src) << "Data : " << nnum1 << ", " << nnum2 << ", " << i;
          }
        }
      }

      // comopare the split label
      if (!parser_batch.info().has_property(OctreeInfo::kSplit)) continue;
      const float* split1 = parser1.split_cpu(d);
      const float* split2 = parser2.split_cpu(d);
      const float* split_batch = parser_batch.split_cpu(d);
      for (int i = 0; i < nnumb; ++i) {
        float src = i < nnum1 ? split1[i] : split2[i - nnum1];
        float des = split_batch[i];
        ASSERT_EQ(des, src) << "Split : " << nnum1 << ", " << nnum2 << ", " << i;
      }
    }
  }

};

TYPED_TEST_CASE(OctreeUtilTest, TestDtypesAndDevices);

TYPED_TEST(OctreeUtilTest, TestOctreeBatch1) {
  // same octrees, ordinary octrees
  test_octree_batch("octree_1", "octree_1");
  test_octree_batch("octree_2", "octree_2");
}

TYPED_TEST(OctreeUtilTest, TestOctreeBatch2) {
  // different octrees, ordinary octrees
  test_octree_batch("octree_2", "octree_1");
  test_octree_batch("octree_1", "octree_2");
}

TYPED_TEST(OctreeUtilTest, TestOctreeBatch3) {
  // same octrees, adaptive octrees
  test_octree_batch("octree_3", "octree_3");
  test_octree_batch("octree_4", "octree_4");
}

TYPED_TEST(OctreeUtilTest, TestOctreeBatch4) {
  // different octrees, adaptive octrees
  test_octree_batch("octree_3", "octree_4");
  test_octree_batch("octree_4", "octree_3");
}

TYPED_TEST(OctreeUtilTest, TestXYZ2Key) {
  const int num = 3, depth = 5;
  unsigned char xyz[] = { 0, 0, 0, 1, 1, 1, 1, 2, 1, 2, 3, 0 };
  unsigned char key[] = { 0, 0, 0, 1, 7, 0, 0, 2, 29, 0, 0, 0 };
  Blob<unsigned int> blob_xyz(vector<int> {3}), blob_key(vector<int> {3});
  memcpy(blob_xyz.mutable_cpu_data(), xyz, sizeof(xyz));
  if (Caffe::mode() == Caffe::CPU) {
    octree::xyz2key_cpu(blob_key.mutable_cpu_data(), blob_xyz.cpu_data(), num, depth);
  } else {
    octree::xyz2key_gpu(blob_key.mutable_gpu_data(), blob_xyz.gpu_data(), num, depth);
  }

  const unsigned int* rst = blob_key.cpu_data();
  unsigned int* rst_gt = reinterpret_cast<unsigned int*>(key);
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(rst[i], rst_gt[i]);
  }
}

} // namespace caffe