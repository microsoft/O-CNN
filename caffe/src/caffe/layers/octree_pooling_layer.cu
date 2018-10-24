#include "caffe/layers/octree_pooling_layer.hpp"
#include "caffe/util/octree.hpp"

namespace caffe {

template <typename Dtype>
__global__ void max_pooling_forward(Dtype* top_data, const int top_h,
    int* bottom_mask, const Dtype* bottom_data,	const int bottom_h, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int h = i % top_h;
    int c = i / top_h;

    int hb = 8 * h;
    int max_idx = hb;
    bottom_data += c * bottom_h;
    Dtype max_val = bottom_data[hb];
#pragma unroll 7
    for (int idx = hb + 1; idx < hb + 8; ++idx) {
      Dtype value = bottom_data[idx];
      if (value > max_val) {
        max_idx = idx;
        max_val = value;
      }
    }

    top_data[i] = max_val;
    bottom_mask[i] = max_idx;
  }
}

template <typename Dtype>
__global__ void max_pooling_backward(Dtype* bottom_diff, const int bottom_h,
    const int* bottom_mask, const Dtype* top_diff, const int top_h, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int c = i / top_h;
    bottom_diff[c * bottom_h + bottom_mask[i]] = top_diff[i];
  }
}

template <typename Dtype>
void OctreePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = top.size() == 2 ?
      (int*)top[1]->mutable_gpu_data() : max_idx_.mutable_gpu_data();
  Dtype* top_data = top_buffer_->mutable_gpu_data();
  const Dtype* btm_data = bottom[0]->gpu_data();

  // pooling
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int top_h = bottom_h / 8;
  int num = top_h * channel;
  if (num != 0) {
    max_pooling_forward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>> (
        top_data, top_h, mask, btm_data, bottom_h, num);
    CUDA_POST_KERNEL_CHECK;
  }

  // pad
  octree::pad_forward_gpu<Dtype>(top[0]->mutable_gpu_data(),
      top[0]->shape(2), top[0]->shape(1), top_buffer_->gpu_data(),
      top_buffer_shape_[2], octree_batch_.children_gpu(curr_depth_ - 1));
}


template <typename Dtype>
void OctreePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  octree::pad_backward_gpu<Dtype>(top_buffer_->mutable_gpu_data(),
      top_buffer_shape_[2], top_buffer_shape_[1], top[0]->gpu_diff(),
      top[0]->shape(2), octree_batch_.children_gpu(curr_depth_ - 1));

  const int* mask = top.size() == 2 ?
      (const int*)top[1]->gpu_data() : max_idx_.gpu_data();
  const Dtype* top_diff = top_buffer_->gpu_data();
  Dtype* btm_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), btm_diff);

  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int top_h = bottom_h / 8;
  int num = top_h * channel;
  if (num == 0) return;
  max_pooling_backward<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>>(
      btm_diff, bottom_h, mask, top_diff, top_h, num);
  CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(OctreePoolingLayer);

}