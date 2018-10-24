
#include "caffe/test/test_octree.hpp"
#include "caffe/layers/octree_conv_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/octree_padding_layer.hpp"
#include "caffe/layers/octree2col_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctreeConvLayerTest : public OctreeTest<TypeParam> {
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

  void LayerForwardBackward(const vector<int>& kernel_size, const int stride) {
    const int curr_depth = 4;
    const int num_output = 3;
    const int channel = 3;
    const char* filler_type = "xavier";
    ASSERT_EQ(3, kernel_size.size());
    const int kernel_sdim = kernel_size[0] * kernel_size[1] * kernel_size[2];

    // reshaepe bottom
    ReshapeBottom(channel, curr_depth);

    /// FORWARD
    // OctreeConv
    LayerParameter layer_param;
    layer_param.mutable_octree_param()->set_curr_depth(curr_depth);
    auto conv_parm = layer_param.mutable_convolution_param();
    conv_parm->set_num_output(num_output);
    conv_parm->mutable_weight_filler()->set_type(filler_type);
    conv_parm->mutable_bias_filler()->set_type(filler_type);
    conv_parm->add_stride(stride);
    for (int i = 0; i < 3; ++i) conv_parm->add_kernel_size(kernel_size[i]);

    OctreeConvLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Octree2Col + reshape + convolution + OctreePad + reshape
    Octree2ColLayer<Dtype> layer_oct2col(layer_param);
    Blob<Dtype> oct2col(vector<int> {1});
    vector<Blob<Dtype>*> blob_top_vec_oct2col{ &oct2col };
    layer_oct2col.SetUp(this->blob_bottom_vec_, blob_top_vec_oct2col);
    layer_oct2col.Forward(this->blob_bottom_vec_, blob_top_vec_oct2col);

    auto shape_param = layer_param.mutable_reshape_param()->mutable_shape();
    shape_param->add_dim(0);
    shape_param->add_dim(channel);
    shape_param->add_dim(kernel_sdim);
    shape_param->add_dim(-1);
    ReshapeLayer<Dtype> layer_reshape(layer_param);
    Blob<Dtype> reshp(vector<int> {1});
    vector<Blob<Dtype>*> blob_top_vec_reshp{ &reshp };
    layer_reshape.SetUp(blob_top_vec_oct2col, blob_top_vec_reshp);
    layer_reshape.Forward(blob_top_vec_oct2col, blob_top_vec_reshp);

    conv_parm->clear_stride();
    conv_parm->clear_kernel_size();
    conv_parm->set_kernel_h(kernel_sdim);
    conv_parm->set_kernel_w(1);
    ConvolutionLayer<Dtype> layer_conv(layer_param);
    Blob<Dtype> conv(vector<int> {1});
    vector<Blob<Dtype>*> blob_top_vec_conv{ &conv };
    layer_conv.SetUp(blob_top_vec_reshp, blob_top_vec_conv);
    // share parameter
    auto& target_blobs = layer_conv.blobs();
    auto& source_blobs = layer.blobs();
    ASSERT_EQ(2, target_blobs.size());
    for (int i = 0; i < target_blobs.size(); ++i) {
      ASSERT_TRUE(source_blobs[i]->shape() == target_blobs[i]->shape());
      target_blobs[i]->ShareData(*source_blobs[i]);
    }
    layer_conv.Forward(blob_top_vec_reshp, blob_top_vec_conv);

    shape_param = layer_param.mutable_reshape_param()->mutable_shape();
    shape_param->clear_dim();
    shape_param->add_dim(0);
    shape_param->add_dim(channel);
    shape_param->add_dim(-1);
    shape_param->add_dim(1);
    ReshapeLayer<Dtype> layer_reshape1(layer_param);
    Blob<Dtype> reshp1(vector<int> {1});
    vector<Blob<Dtype>*> blob_top_vec_reshp1{ &reshp1 };
    layer_reshape1.SetUp(blob_top_vec_conv, blob_top_vec_reshp1);
    layer_reshape1.Forward(blob_top_vec_conv, blob_top_vec_reshp1);

    Blob<Dtype> octree_pad;
    if (stride == 2)
      layer_param.mutable_octree_param()->set_curr_depth(curr_depth - 1);
    OctreePaddingLayer<Dtype> layer_pad(layer_param);
    vector<Blob<Dtype>*> blob_top_vec_pad{ &octree_pad };
    layer_pad.SetUp(blob_top_vec_reshp1, blob_top_vec_pad);
    layer_pad.Forward(blob_top_vec_reshp1, blob_top_vec_pad);

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

    // Octree2Col + reshape + convolution backward
    octree_pad.ShareDiff(*blob_top);
    layer_pad.Backward(blob_top_vec_pad, propagrate_down, blob_top_vec_reshp1);
    layer_reshape1.Backward(blob_top_vec_reshp1, propagrate_down, blob_top_vec_conv);
    layer_conv.Backward(blob_top_vec_conv, propagrate_down, blob_top_vec_reshp);
    layer_reshape.Backward(blob_top_vec_reshp, propagrate_down, blob_top_vec_oct2col);
    Blob<Dtype> bottom_ref;
    bottom_ref.ReshapeLike(*blob_bottom);
    bottom_ref.ShareData(*blob_bottom);
    vector<Blob<Dtype>*> blob_bottom_ref{ &bottom_ref };
    layer_oct2col.Backward(blob_top_vec_oct2col, propagrate_down, blob_bottom_ref);

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

    OctreeConvLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
};

TYPED_TEST_CASE(OctreeConvLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctreeConvLayerTest, TestForwardBackwardFunc) {
  vector<int> stride{ 1, 2 };
  vector<vector<int> > kernel_size{ { 3, 3, 3 }, { 2, 2, 2 }, { 3, 1, 1 },
    { 3, 3, 1 }, { 1, 1, 1 } };

  for (int i = 0; i < stride.size(); ++i) {
    for (int j = 0; j < kernel_size.size(); ++j) {
      this->LayerForwardBackward(kernel_size[j], stride[i]);
    }
  }
}

TYPED_TEST(OctreeConvLayerTest, TestBackwardFunc) {
  vector<int> stride{ 1, 2 };
  vector<vector<int> > kernel_size{ { 3, 3, 3 }, { 2, 2, 2 }, { 3, 1, 1 },
    { 3, 3, 1 }, { 1, 1, 1 } };

  for (int i = 0; i < stride.size(); ++i) {
    for (int j = 0; j < kernel_size.size(); ++j) {
      this->LayerBackwardGradChecker(kernel_size[j], stride[i]);
    }
  }

}
}