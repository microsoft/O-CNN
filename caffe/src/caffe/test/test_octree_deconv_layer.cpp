
#include "caffe/test/test_octree.hpp"
#include "caffe/layers/octree_deconv_layer.hpp"
#include "caffe/layers/octree_conv_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/octree_depadding_layer.hpp"
#include "caffe/layers/octree_padding_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctreeDeconvLayerTest : public OctreeTest<TypeParam> {
 protected:
  typedef typename TypeParam::Dtype Dtype;

  virtual void SetUp() {
    // load test data
    this->load_test_data(vector<string> {"octree_1", "octree_2"});
    this->set_octree_batch();

    // octree workspace
    //Octree::set_workspace_maxsize(1024);
  }

  void ReshapeBottom(const int channel, const int curr_depth) {
    // fill bottom blob with random data
    Blob<Dtype>& the_octree = Octree::get_octree(Dtype(0));
    this->octree_parser_.set_cpu(the_octree.cpu_data());
    int nnum = this->octree_parser_.info().node_num(curr_depth);
    this->blob_bottom_->Reshape(vector<int> {1, channel, nnum, 1});

    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
  }


  void LayerBackwardGradChecker(const vector<int>& kernel_size, const int stride) {
    const int curr_depth = 1;
    const int num_output = 2;
    const int channel = 2;
    const char* filler_type = "xavier";
    ASSERT_EQ(3, kernel_size.size());

    // reshaepe bottom
    ReshapeBottom(channel, curr_depth);

    // param
    LayerParameter layer_param;
    layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
    auto conv_parm = layer_param.mutable_convolution_param();
    conv_parm->set_num_output(num_output);
    conv_parm->add_stride(stride);
    conv_parm->mutable_weight_filler()->set_type(filler_type);
    conv_parm->mutable_bias_filler()->set_type(filler_type);
    for (int i = 0; i < 3; ++i) conv_parm->add_kernel_size(kernel_size[i]);

    OctreeDeconvLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
};

TYPED_TEST_CASE(OctreeDeconvLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctreeDeconvLayerTest, TestForwardBackwardFunc) {
  const int curr_depth = 4;
  const int num_output = 3;
  const int channel = 3;
  const int stride = 2; // only for kernel_size==2 and stride==2
  const vector<int> kernel_size{ 2, 2, 2 };
  const char* filler_type = "xavier";
  ASSERT_EQ(3, kernel_size.size());
  const int kernel_sdim = kernel_size[0] * kernel_size[1] * kernel_size[2];

  // reshaepe bottom
  ReshapeBottom(channel, curr_depth);

  /// FORWARD
  // OctreeDeconv
  LayerParameter layer_param;
  layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
  auto conv_parm = layer_param.mutable_convolution_param();
  conv_parm->set_num_output(num_output);
  conv_parm->add_stride(stride);
  conv_parm->mutable_weight_filler()->set_type(filler_type);
  conv_parm->mutable_bias_filler()->set_type(filler_type);
  for (int i = 0; i < 3; ++i) conv_parm->add_kernel_size(kernel_size[i]);

  OctreeDeconvLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // OctreeDepad + convolution + OctreePad
  OctreeDepaddingLayer<Dtype> layer_depad(layer_param);
  Blob<Dtype> octdepad(vector<int> {1});
  vector<Blob<Dtype>*> blob_top_vec_depad{ &octdepad };
  layer_depad.SetUp(this->blob_bottom_vec_, blob_top_vec_depad);
  layer_depad.Forward(this->blob_bottom_vec_, blob_top_vec_depad);

  conv_parm->clear_kernel_size();
  conv_parm->set_kernel_h(kernel_sdim);
  conv_parm->set_kernel_w(1);
  conv_parm->clear_stride();
  conv_parm->set_stride_h(kernel_sdim);
  conv_parm->set_stride_w(1);
  DeconvolutionLayer<Dtype> layer_deconv(layer_param);
  Blob<Dtype> deconv(vector<int> {1});
  vector<Blob<Dtype>*> blob_top_vec_conv{ &deconv };
  layer_deconv.SetUp(blob_top_vec_depad, blob_top_vec_conv);

  // share parameter
  auto& target_blobs = layer_deconv.blobs();
  auto& source_blobs = layer.blobs();
  ASSERT_EQ(2, target_blobs.size());
  for (int i = 0; i < target_blobs.size(); ++i) {
    ASSERT_TRUE(source_blobs[i]->shape() == target_blobs[i]->shape());
    target_blobs[i]->ShareData(*source_blobs[i]);
  }
  layer_deconv.Forward(blob_top_vec_depad, blob_top_vec_conv);

  Blob<Dtype> octree_pad;
  layer_param.mutable_octree_param()->set_curr_depth(curr_depth + 1);
  OctreePaddingLayer<Dtype> layer_pad(layer_param);
  vector<Blob<Dtype>*> blob_top_vec_pad{ &octree_pad };
  layer_pad.SetUp(blob_top_vec_conv, blob_top_vec_pad);
  layer_pad.Forward(blob_top_vec_conv, blob_top_vec_pad);

  // check Forward
  const Dtype* top_data_octconv = this->blob_top_->cpu_data();
  const Dtype* top_data_conv = octree_pad.cpu_data();
  ASSERT_EQ(octree_pad.count(), this->blob_top_->count());
  for (int i = 0; i < octree_pad.count(); ++i) {
    ASSERT_FLOAT_EQ(top_data_conv[i], top_data_octconv[i]);
  }


  ///// BACKWARD
  // fill the top_diff
  auto blob_top = this->blob_top_;
  auto blob_bottom = this->blob_bottom_;
  caffe_rng_uniform<Dtype>(blob_top->count(), -1.0, 1.0, blob_top->mutable_cpu_diff());

  // OctreeConv backward
  vector<bool> propagrate_down{ true };
  layer.Backward(this->blob_top_vec_, propagrate_down, this->blob_bottom_vec_);

  octree_pad.ShareDiff(*blob_top);
  layer_pad.Backward(blob_top_vec_pad, propagrate_down, blob_top_vec_conv);
  layer_deconv.Backward(blob_top_vec_conv, propagrate_down, blob_top_vec_depad);
  Blob<Dtype> bottom_ref;
  bottom_ref.ReshapeLike(*blob_bottom);
  bottom_ref.ShareData(*blob_bottom);
  vector<Blob<Dtype>*> blob_bottom_ref{ &bottom_ref };
  layer_depad.Backward(blob_top_vec_depad, propagrate_down, blob_bottom_ref);

  // check Backward
  const Dtype* bottom_diff = blob_bottom->cpu_diff();
  const Dtype* bottom_diff_ref = bottom_ref.cpu_diff();
  for (int i = 0; i < bottom_ref.count(); ++i) {
    ASSERT_NEAR(bottom_diff_ref[i], bottom_diff[i], 1.0e-5);
  }
  for (int n = 0; n < target_blobs.size(); ++n) {
    const Dtype* blob_diff = target_blobs[n]->cpu_diff();
    const Dtype* blob_diff_ref = source_blobs[n]->cpu_diff();
    for (int i = 0; i < source_blobs[n]->count(); ++i) {
      ASSERT_NEAR(blob_diff_ref[i], blob_diff[i], 1.0e-5);
    }
  }
}

TYPED_TEST(OctreeDeconvLayerTest, TestBackwardFunc) {
  vector<int> stride{ 1, 2 };
  vector<vector<int> > kernel_size{ { 3, 3, 3 }, /*{ 2, 2, 2 },*/ { 3, 1, 1 },
    /* { 3, 3, 1 }, */{ 1, 1, 1 } };

  for (int i = 0; i < stride.size(); ++i) {
    for (int j = 0; j < kernel_size.size(); ++j) {
      this->LayerBackwardGradChecker(kernel_size[j], stride[i]);
    }
  }

}
}