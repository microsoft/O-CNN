#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

#define sign(x) ((Dtype(0) < (x)) - ((x) < Dtype(0)))

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  top[0]->Reshape(bottom_shape);
  squared_.Reshape(bottom_shape);
  bottom_shape[1] = 1;
  norm_.Reshape(bottom_shape);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* square_data = squared_.mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int spatial_dim = bottom[0]->count(2);
  caffe_sqr<Dtype>(num * channels * spatial_dim, bottom_data, square_data);
  for (int n = 0; n < num; n++) {
    for (int s = 0; s < spatial_dim; s++) {
      int i = n * spatial_dim + s;
      norm_data[i] = Dtype(0);
      for (int c = 0; c < channels; c++) {
        int j = (n * channels + c) * spatial_dim + s;
        norm_data[i] += square_data[j];
      }
      norm_data[i] += eps_; // avoid dividing by zero
      norm_data[i] = sqrt(norm_data[i]);
      for (int c = 0; c < channels; c++) {
        int j = (n * channels + c) * spatial_dim + s;
        top_data[j] = bottom_data[j] / norm_data[i];
      }
    }
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) return;

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int spatial_dim = bottom[0]->count(2);
  for (int n = 0; n < num; ++n) {
    for (int s = 0; s < spatial_dim; s++) {
      int i = n * spatial_dim + s;
      int k = n * channels * spatial_dim + s;
      Dtype a = caffe_cpu_strided_dot(channels, top_data + k,  spatial_dim, top_diff + k, spatial_dim);
      for (int c = 0; c < channels; c++) {
        int j = (n * channels + c) * spatial_dim + s;
        bottom_diff[j] = (top_diff[j] - top_data[j] * a) / norm_data[i];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
