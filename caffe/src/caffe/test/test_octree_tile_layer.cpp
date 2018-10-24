
#include "caffe/test/test_octree.hpp"
#include "caffe/layers/octree_tile_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctreeTileLayerTest : public OctreeTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctreeTileLayerTest() : test_depth_(3), test_channel_(6) {}

  virtual void SetUp() {
    // load test data
    this->load_test_data(vector<string> {"octree_1"});
    this->set_octree_batch();

    // fill bottom blob with random data
    Blob<Dtype>& the_octree = Octree::get_octree(Dtype(0));
    this->octree_parser_.set_cpu(the_octree.cpu_data());
    int nnum = this->octree_parser_.info().node_num(test_depth_);
    this->blob_bottom_->Reshape(vector<int> {1, test_channel_, nnum, 1});

    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
  }

 protected:
  const int test_channel_;
  const int test_depth_;
};

TYPED_TEST_CASE(OctreeTileLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctreeTileLayerTest, TestForwardFunc) {
  typedef typename TypeParam::Dtype Dtype;
  const int curr_depth = this->test_depth_;
  const int tile_depth = curr_depth + 2;
  LayerParameter layer_param;
  layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
  layer_param.mutable_octree_param()->set_tile_depth(tile_depth);

  OctreeTileLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // find the non-empty node
  OctreeParser& octree_parser = this->octree_parser_;
  int nnum = octree_parser.info().node_num(curr_depth);
  const int* children = octree_parser.children_cpu(curr_depth);
  int idx = 0;
  while (children[idx] == -1) idx++;

  // check
  int top_h = this->blob_top_->shape(2);
  int channel = this->blob_top_->shape(1);
  int bottom_h = this->blob_bottom_->shape(2);
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int c = 0; c < channel; ++c) {
    for (int h = 0; h < top_h; ++h) {
      EXPECT_EQ(top_data[c*top_h + h], bottom_data[c*bottom_h + idx]);
    }
  }
  SUCCEED() << "Early message!";
}

TYPED_TEST(OctreeTileLayerTest, TestBackwardFunc) {
  typedef typename TypeParam::Dtype Dtype;
  const int curr_depth = this->test_depth_;
  const int tile_depth = curr_depth + 2;
  LayerParameter layer_param;
  layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
  layer_param.mutable_octree_param()->set_tile_depth(tile_depth);
  OctreeTileLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
}