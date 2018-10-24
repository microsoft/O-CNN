#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/argmax_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void argmax_forward(Dtype* top_data, const Dtype* bottom_data,
    const int dim, const int axis_dist, const int num) {
  CUDA_KERNEL_LOOP(i, num) {
    int idx = i / axis_dist * dim * axis_dist + (i % axis_dist);

    Dtype data = 0;
    Dtype max_val = bottom_data[idx];
//#pragma unroll 6
    for (int j = 1; j < dim; ++j) {
      Dtype val = bottom_data[idx + j * axis_dist];
      if (val > max_val) {
        max_val = val;
        data = j;
      }
    }

    top_data[i] = data;
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // todo: totally replace the following function with the GPU implentation
  if (!has_axis_ || top_k_ != 1 || out_max_val_) {
    ArgMaxLayer<Dtype>::Forward_cpu(bottom, top);
    return;
  }

  int dim = bottom[0]->shape(axis_);              // dim = C
  int num = bottom[0]->count() / dim;             // num = N*H*W
  int axis_dist = bottom[0]->count(axis_) / dim;  // dist = H*W
  Dtype* top_data = top[0]->mutable_gpu_data();
  //Dtype* max_val = top[0]->mutable_gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  argmax_forward<Dtype> <<< CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>> (
      top_data, bottom_data, dim, axis_dist, num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ArgMaxLayer);
}  // namespace caffe
