#ifndef CAFFE_OCTREE_UNPOOLING_LAYER_HPP_
#define CAFFE_OCTREE_UNPOOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/octree.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Note: currently only max-unpooling with stride 2 and kernel 2 is supported
// bottom[0]: data; bottom[1]: unpooling mask; top[0]: unpooled data
template <typename Dtype>
class OctreeUnpoolingLayer : public Layer<Dtype> {
 public:
  explicit OctreeUnpoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctreeUnpooling"; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

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
  OctreeParser octree_batch_;
  shared_ptr<Blob<Dtype> > buffer_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_UNPOOLING_LAYER_HPP_
