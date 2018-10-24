#ifndef CAFFE_OCTREE_DATABASE_LAYER_HPP_
#define CAFFE_OCTREE_DATABASE_LAYER_HPP_

#include "caffe/layers/data_layer.hpp"

namespace caffe {

template <typename Dtype>
class OctreeDataBaseLayer : public DataLayer<Dtype> {
 public:
  explicit OctreeDataBaseLayer(const LayerParameter& param);
  virtual ~OctreeDataBaseLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "OctreeDataBase"; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  int RandDropDepth();

 protected:
  // extract the feature from the deepest octree nodes
  shared_ptr<Layer<Dtype> > feature_layer_;
  vector<Blob<Dtype>*> feature_btm_vec_;
  vector<Blob<Dtype>*> feature_top_vec_;

  // parameters
  int batch_size_;
  int curr_depth_;
  int signal_channel_;
  bool output_octree_;
  unsigned int rand_skip_;

  vector<vector<char> > octree_buffer_;

  // dropout
  //bool dropout_;
  vector<int> dropout_depth_;
  vector<float> dropout_ratio_;
};

}  // namespace caffe

#endif  // CAFFE_OCTREE_DATABASE_LAYER_HPP_
