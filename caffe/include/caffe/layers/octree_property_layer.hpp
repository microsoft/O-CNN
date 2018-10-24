#ifndef CAFFE_OCTREE_PROPERTY_LAYER_HPP_
#define CAFFE_OCTREE_PROPERTY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/octree.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OctreePropertyLayer : public Layer<Dtype> {
 public:
  explicit OctreePropertyLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctreeProperty"; }
  virtual inline int MaxTopBlobs() const { return OctreeInfo::kPTypeNum; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;

  bool IsDataProperty(OctreeInfo::PropType ptype);

 protected:
  int curr_depth_;
  int content_flag_;
  int signal_channel_;
  vector<OctreeInfo::PropType> ptypes_;
  OctreeParser octree_batch_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_PROPERTY_LAYER_HPP_
