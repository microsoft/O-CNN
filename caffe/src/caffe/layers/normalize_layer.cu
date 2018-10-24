#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, Dtype epsilon, const Dtype* data, Dtype* norm_data) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    norm_data[index] = sum + epsilon;
  }
}

template <typename Dtype>
__global__ void kernel_channel_scale(const int num, const int channels,
    const int spatial_dim, Dtype alpha, const Dtype* data, const Dtype* norm_data,
    Dtype beta, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    output_data[index] = alpha * data[index] * norm_data[n * spatial_dim + s] + beta * output_data[index];
  }
}

template <typename Dtype>
__global__ void kernel_channel_self_scale(const int num, const int channels, const int spatial_dim,
    const Dtype* norm_data, Dtype* input_output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    input_output_data[index] *= norm_data[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int num, const int channels, const int spatial_dim,
    Dtype alpha, const Dtype* data, const Dtype* norm_data,
    Dtype beta, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    output_data[index] = alpha * data[index] / norm_data[n * spatial_dim + s] + beta * output_data[index];
  }
}

template <typename Dtype>
__global__ void kernel_channel_self_div(const int num, const int channels, const int spatial_dim,
    const Dtype* norm_data, Dtype* input_output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    input_output_data[index] /= norm_data[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
              * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* square_data = squared_.mutable_gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int spatial_dim = bottom[0]->count(2);
  caffe_gpu_powx(num * channels * spatial_dim, bottom_data, Dtype(2), square_data);
  kernel_channel_sum<Dtype> <<< CAFFE_GET_BLOCKS(num*spatial_dim), CAFFE_CUDA_NUM_THREADS >>> (
      num, channels, spatial_dim, eps_, square_data, norm_data);
  caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(0.5), norm_data);
  kernel_channel_div<Dtype> <<< CAFFE_GET_BLOCKS(num*channels*spatial_dim), CAFFE_CUDA_NUM_THREADS >>> (
      num, channels, spatial_dim, Dtype(1), bottom_data, norm_data, Dtype(0), top_data);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) return;

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* norm_data = norm_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* temp_diff = norm_.mutable_gpu_diff();

  int num = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int spatial_dim = bottom[0]->count(2);
  int count = bottom[0]->count();

  kernel_channel_dot<Dtype> <<< CAFFE_GET_BLOCKS(num * spatial_dim), CAFFE_CUDA_NUM_THREADS >>> (
      num, channels, spatial_dim, top_data, top_diff, temp_diff);
  kernel_channel_scale<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>> (
      num, channels, spatial_dim, Dtype(1), top_data, temp_diff, Dtype(0), bottom_diff);
  caffe_gpu_sub(count, top_diff, bottom_diff, bottom_diff);
  kernel_channel_self_div<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>> (
      num, channels, spatial_dim, norm_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);
}  // namespace caffe