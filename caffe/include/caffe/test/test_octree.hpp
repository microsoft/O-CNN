#ifndef CAFFE_TEST_TEST_OCTREE_HPP_
#define CAFFE_TEST_TEST_OCTREE_HPP_

#include <vector>
#include <fstream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/octree.hpp"
#include "caffe/layers/base_data_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ifstream;

namespace testing {
namespace internal {
enum GTestColor { COLOR_DEFAULT, COLOR_RED,	COLOR_GREEN, COLOR_YELLOW };
extern void ColoredPrintf(GTestColor color, const char* fmt, ...);
}  // namespace internal
}  // namespace testing

#define PRINTF(...)  do {                                                            \
	testing::internal::ColoredPrintf(testing::internal::COLOR_GREEN, "[          ] "); \
	testing::internal::ColoredPrintf(testing::internal::COLOR_YELLOW, __VA_ARGS__);    \
	} while(false)

// C++ stream interface
class TestCout : public std::stringstream {
 public:
  ~TestCout() { PRINTF("%s\n", str().c_str()); }
};

#define TEST_COUT  TestCout()

////////////////////////


namespace caffe {

const char *get_test_octree(const char *name, size_t *size = nullptr);

template <typename TypeParam>
class OctreeTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctreeTest() : blob_bottom_(new Blob<Dtype>(vector<int> {8})),
             blob_top_(new Blob<Dtype>(vector<int> {8})),
  octree_batch_(new Batch<Dtype>()) {
    // fix random seed
    Caffe::set_random_seed(1701);

    // set bottom & top blob
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  void load_test_data(vector<string>& data_names) {
    int num = data_names.size();
    octree_buffer_.resize(num);
    for (int i = 0; i < num; ++i) {
      size_t sz = 0;
      const char* ptr_octree = get_test_octree(data_names[i].c_str(), &sz);
      ASSERT_FALSE(ptr_octree == nullptr);

      octree_buffer_[i].resize(sz);
      memcpy(octree_buffer_[i].data(), ptr_octree, sz);
    }
  }

  void set_octree_batch(int signal_channel = 3, int signal_location = 0) {
    // set octree batch
    const int content_flag = 7;
    int num = octree_buffer_.size();
    octree::merge_octrees(octree_batch_->data_, octree_buffer_);
    Blob<Dtype>& the_octree = Octree::get_octree(Dtype(0));
    the_octree.ReshapeLike(octree_batch_->data_);
    the_octree.set_cpu_data(octree_batch_->data_.mutable_cpu_data());

    // set octree parser
    octree_parser_.set_cpu(the_octree.cpu_data());
    if (Caffe::mode() == Caffe::GPU)
      octree_parser_.set_gpu(the_octree.gpu_data());
  }

  void load_octree(vector<char>& octree, const string filename) {
    ifstream infile(filename, std::ifstream::binary);
    ASSERT_FALSE(infile.fail()) << "Cannot open the test file " << filename;
    infile.seekg(0, infile.end);
    size_t len = infile.tellg();
    infile.seekg(0, infile.beg);
    octree.resize(len);
    infile.read(octree.data(), len);
    infile.close();
  }

  void blob2string(string& str, Blob<Dtype>& blob) {
    str.clear();
    const Dtype* blob_data = blob.cpu_data();
    for (int i = 0; i < blob.count(); ++i) {
      str += std::to_string(blob_data[i]) + ", ";
    }
  }

  virtual ~OctreeTest() {
    delete octree_batch_;
    delete blob_bottom_;
    delete blob_top_;
  }

 protected:
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Batch<Dtype>* const octree_batch_;
  OctreeParser octree_parser_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  vector<vector<char> > octree_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_OCTREE_HPP_