#ifndef CAFFE_OCTREE_SET_FEATURE_LAYER_HPP_
#define CAFFE_OCTREE_SET_FEATURE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/octree.hpp"

namespace caffe {

template <typename Dtype>
class OctreeSetFeatureLayer : public Layer<Dtype> {
 public:
  explicit OctreeSetFeatureLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctreeSetFeature"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  int curr_depth_;
  int adap_depth_;
  bool is_adaptive_;
  int channel_;
  int location_;
  OctreeInfo oct_info_;
  OctreeParser oct_parser_btm_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_SET_FEATURE_LAYER_HPP_
