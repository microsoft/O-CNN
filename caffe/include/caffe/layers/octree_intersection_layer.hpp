#ifndef CAFFE_OCTREE_INTERSECTION_LAYER_HPP_
#define CAFFE_OCTREE_INTERSECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/octree.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OctreeIntersectionLayer : public Layer<Dtype> {
 public:
  explicit OctreeIntersectionLayer(const LayerParameter& param);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctreeIntersection"; }
  // bottom: octree_gt, octree_new, label
  virtual inline int ExactNumBottomBlobs() const override { return 4; }
  virtual inline int ExactNumTopBlobs() const override { return 2; }

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
  Blob<int> index_;
  Blob<int> index_gt_;

  // these are just buffers
  Blob<int> index_all_;
  Blob<unsigned> key_intersection_;
  Blob<unsigned> shuffled_key_;
  Blob<unsigned> shuffled_key_gt_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_INTERSECTION_LAYER_HPP_
