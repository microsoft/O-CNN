#include "caffe/layers/octree2col_layer.hpp"
#include "caffe/test/test_octree.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Octree2ColTest : public OctreeTest<TypeParam> {
 protected:
  typedef typename TypeParam::Dtype Dtype;

  Octree2ColTest() : channel_(4), curr_depth_(1) {}

  virtual void SetUp() {
    // load test data
    this->load_test_data(vector<string> {"octree_1"});
    this->set_octree_batch();

    // fill bottom blob with random data
    Blob<Dtype>& the_octree = Octree::get_octree(Dtype(0));
    this->octree_parser_.set_cpu(the_octree.cpu_data());
    int nnum = this->octree_parser_.info().node_num(curr_depth_);
    this->blob_bottom_->Reshape(vector<int> {1, channel_, nnum, 1});

    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);

    // index map
    vec_ = vector<vector<int> > { {} /* 333 */, { 13 } /* 111 */,
      { 13, 14, 16, 17, 22, 23, 25, 26 } /* 222 */,
      { 4, 13, 22 } /* 311 */,
      { 10, 13, 16 } /* 131 */,
      { 12, 13, 14 } /* 113 */,
      { 1,  4,  7, 10, 13, 16, 19, 22, 25 } /* 331 */,
      { 3,  4,  5, 12, 13, 14, 21, 22, 23 } /* 313 */,
      { 9, 10, 11, 12, 13, 14, 15, 16, 17 } /* 133 */
    };
    for (int i = 0; i < 27; ++i)  vec_[0].push_back(i);
  }

  //!!! NOTE: this function just tests the 1st level of the octree
  void LayerForward(const vector<int>& kernel_size, const int stride,
      const vector<int>& idx_map) {
    typedef unsigned char ubyte;

    char msg[256];
    sprintf(msg, "kernel_size: %d%d%d, stride: %d",
        kernel_size[0], kernel_size[1], kernel_size[2], stride);

    LayerParameter layer_param;
    layer_param.mutable_octree_param()->set_curr_depth(curr_depth_);
    ASSERT_EQ(3, kernel_size.size());
    for (int i = 0; i < 3; ++i) {
      layer_param.mutable_convolution_param()->add_kernel_size(kernel_size[i]);
    }

    Octree2ColLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    int top_h = this->blob_top_->shape(2);
    int kernel = kernel_size[0] * kernel_size[1] * kernel_size[2];
    ASSERT_EQ(idx_map.size(), kernel);
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    for (int c = 0; c < this->channel_; ++c) {
      for (int k = 0; k < kernel; ++k) {
        for (int h = 0; h < top_h; ++h) {
          ubyte z = h & 1;
          ubyte y = (h & 2) >> 1;
          ubyte x = h >> 2;

          int kmap = idx_map[k];
          ubyte dz = kmap % 3, t = kmap / 3;
          ubyte dy = t % 3,   dx = t / 3;

          z = z + dz - 1;
          y = y + dy - 1;
          x = x + dx - 1;

          Dtype val = 0;
          if (z < 2 && y < 2 && x < 2) {
            val = bottom_data[c * top_h + x * 4 + y * 2 + z];
          }
          ASSERT_EQ(val, top_data[(c * kernel + k)*top_h + h]) << msg;
        }
      }
    }
  }


  void LayerBackward(const vector<int>& kernel_size, const int stride,
      const vector<int>& idx_map) {
    char msg[256];
    sprintf(msg, "kernel_size: %d%d%d, stride: %d",
        kernel_size[0], kernel_size[1], kernel_size[2], stride);
    //TEST_COUT << msg;

    LayerParameter layer_param;
    layer_param.mutable_octree_param()->set_curr_depth(curr_depth_);
    ASSERT_EQ(3, kernel_size.size());
    for (int i = 0; i < 3; ++i)	{
      layer_param.mutable_convolution_param()->add_kernel_size(kernel_size[i]);
    }

    Octree2ColLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

 protected:
  const int channel_;
  const int curr_depth_;
  Blob<Dtype> data_octree_;
  Blob<Dtype> data_col_;
  vector<vector<int> > vec_;
};

TYPED_TEST_CASE(Octree2ColTest, TestDtypesAndDevices);

TYPED_TEST(Octree2ColTest, TestForwardFun) {
  vector<int> stride{ 1, 2 };
  vector<int> vi{ 0, 2, 3, 6, 1 };
  vector<vector<int> > kernel_size{
    { 3, 3, 3 }, { 2, 2, 2 }, { 3, 1, 1 }, { 3, 3, 1 }, { 1, 1, 1 }
  };

  for (int i = 0; i < stride.size(); ++i) {
    for (int j = 0; j < vi.size(); ++j) {
      this->LayerForward(kernel_size[j], stride[i], this->vec_[vi[j]]);
    }
  }
}

TYPED_TEST(Octree2ColTest, TestBackwardFun) {
  vector<int> stride{ 1, 2 };
  vector<int> vi{ 0,/* 2, 3,*/ 6, 1 };
  vector<vector<int> > kernel_size{ { 3, 3, 3 }, /*{ 2, 2, 2 },{ 3, 1, 1 },*/
    { 3, 3, 1 }, { 1, 1, 1 } };

  for (int i = 0; i < stride.size(); ++i) {
    for (int j = 0; j < vi.size(); ++j) {
      this->LayerBackward(kernel_size[j], stride[i], this->vec_[vi[j]]);
    }
  }
}
}