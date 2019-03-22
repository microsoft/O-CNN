#ifndef CAFFE_ACCURACY_BINARY_LAYER_HPP_
#define CAFFE_ACCURACY_BINARY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
* @brief Computes the classification accuracy for a binary classification task.
* The first top blob contains classification accuracy, the second top blob 
* contains percentage of ground-truth label and prediction label pairs : 
* 0-0, 0-1, 1-0 and 1-1 
*/
template <typename Dtype>
class AccuracyBinaryLayer : public Layer<Dtype> {
 public:
  explicit AccuracyBinaryLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AccuracyBinary"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2;	}
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  Blob<int> buffer_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_BINARY_LAYER_HPP_
