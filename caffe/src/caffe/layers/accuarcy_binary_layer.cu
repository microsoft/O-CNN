#include "caffe/layers/accuracy_binary_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace caffe {

template <typename Dtype>
__global__ void accuracy_forward_kernel(int* buffer, const Dtype* bottom_data,
    const Dtype* bottom_label, const int n) {
  CUDA_KERNEL_LOOP(i, n) {
    int label_gt = static_cast<int>(bottom_label[i]);
    int label = static_cast<int>(bottom_data[i] > 0);

    buffer[i] = label_gt * 2 + label;
  }
}

template <typename Dtype>
void AccuracyBinaryLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  buffer_.Reshape(bottom[0]->shape());
  int* buffer = buffer_.mutable_gpu_data();
  int count = bottom[0]->count();
  accuracy_forward_kernel <Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>> (
      buffer_.mutable_gpu_data(), bottom_data, bottom_label, count);

  int num[4] = { 0 };
  for (int i = 0; i < 4; ++i) {
    num[i] = thrust::count(thrust::device, buffer, buffer + count, i);
  }

  top[0]->mutable_cpu_data()[0] = Dtype(num[0] + num[3]) / Dtype(count);
  if (top.size() == 2) {
    Dtype* top_data = top[1]->mutable_cpu_data();
    for (int i = 0; i < 4; ++i) {
      top_data[i] = Dtype(num[i]) / Dtype(count);
    }
  }
}

template <typename Dtype>
void AccuracyBinaryLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) NOT_IMPLEMENTED;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AccuracyBinaryLayer);

}  // namespace caffe
