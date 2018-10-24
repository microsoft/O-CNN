#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template <typename TypeParam>
class NormalizationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NormalizationLayerTest()
    : epsilon_(Dtype(1e-6)),
      blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormalizationLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  void ReferenceForward(Blob<Dtype>* blob_top, const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void NormalizationLayerTest<TypeParam>::ReferenceForward(Blob<Dtype>* blob_top,
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype epsilon = 1.0e-30;

  vector<int> bottom_shape = blob_bottom.shape();
  int num = bottom_shape[0];
  int channel = bottom_shape[1];
  int spatial_dim = bottom_shape[2] * bottom_shape[3];

  blob_top->Reshape(bottom_shape);
  Dtype* top_data = blob_top->mutable_cpu_data();
  const Dtype* bottom_data = blob_bottom.cpu_data();

  for (int n = 0; n < num; ++n) {
    for (int s = 0; s < spatial_dim; ++s) {
      Dtype norm = epsilon;
      for (int c = 0; c < channel; ++c) {
        int i = (n * channel + c) * spatial_dim + s;
        norm += bottom_data[i] * bottom_data[i];
      }
      norm = sqrt(norm);
      for (int c = 0; c < channel; ++c) {
        int i = (n * channel + c) * spatial_dim + s;
        top_data[i] = bottom_data[i] / norm;
      }
    }
  }
}

TYPED_TEST_CASE(NormalizationLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizationLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceForward(&top_reference, *(this->blob_bottom_), layer_param);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
        this->epsilon_);
  }
}

TYPED_TEST(NormalizationLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
