#ifndef CAFFE_OCTREE_POOLING_LAYER_HPP_
#define CAFFE_OCTREE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/octree.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Note: currently only max-pooling with stride 2 and kernel 2 is supported
// bottom[0]: data; top[0]: pooled data; top[1]: max pooling index, optional
template <typename Dtype>
class OctreePoolingLayer : public Layer<Dtype> {
 public:
  explicit OctreePoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctreePooling"; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

 protected:
  int curr_depth_;
  Blob<int> max_idx_;
  //Blob<Dtype> top_buffer_;
  OctreeParser octree_batch_;
  vector<int> top_buffer_shape_;
  shared_ptr<Blob<Dtype> > top_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_POOLING_LAYER_HPP_
