#ifndef CAFFE_OCTREE_CONV_LAYER_HPP_
#define CAFFE_OCTREE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/octree_base_conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/octree.hpp"

namespace caffe {

template <typename Dtype>
class OctreeConvLayer : public OctreeBaseConvLayer<Dtype> {
 public:
  explicit OctreeConvLayer(const LayerParameter& param)
    : OctreeBaseConvLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "OctreeConv"; }

 protected:
  virtual inline bool is_deconvolution_layer() { return false; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};
}  // namespace caffe

#endif  // CAFFE_OCTREE_CONV_LAYER_HPP_
