#ifndef CAFFE_OCTREE_GROW_LAYER_HPP_
#define CAFFE_OCTREE_GROW_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/octree.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OctreeGrowLayer : public Layer<Dtype> {
 public:
  explicit OctreeGrowLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctreeGrow"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int MaxBottomBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  void update_octreeinfo();

 protected:
  int curr_depth_;
  int batch_size_;
  int node_num_;
  bool full_octree_;
  bool full_octree_init_;
  OctreeInfo oct_info_;
  OctreeParser oct_parser_btm_;

 private:
  static int node_num_capacity_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_GROW_LAYER_HPP_
