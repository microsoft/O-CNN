#ifndef CAFFE_OCTREE_FULL_VOXEL_LAYER_HPP_
#define CAFFE_OCTREE_FULL_VOXEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/octree.hpp"

namespace caffe {

template <typename Dtype>
class Octree2FullVoxelLayer : public Layer<Dtype> {
 public:
  explicit Octree2FullVoxelLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  virtual inline const char* type() const { return "Octree2FullVoxel"; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinBottomBlobs() const { return 1; }
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

 private:
  void build_mapping(int depth);

 protected:
  int curr_depth_;
  int batch_size_;
  OctreeParser octree_batch_;
  Blob<unsigned int> index_mapper_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_FULL_VOXEL_LAYER_HPP_
