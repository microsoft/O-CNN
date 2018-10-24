#ifndef CAFFE_OCTREE2COL_LAYER_HPP_
#define CAFFE_OCTREE2COL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/octree.hpp"

namespace caffe {

template <typename Dtype>
class Octree2ColLayer : public Layer<Dtype> {
 public:
  explicit Octree2ColLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  virtual inline const char* type() const { return "Octree2Col"; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  int stride_;
  vector<int> kernel_size_;
  int kernel_dim_;
  int kernel_sdim_;
  int channels_;
  int curr_depth_;
  bool is_1x1_;
  OctreeParser octree_batch_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE2COL_LAYER_HPP_
