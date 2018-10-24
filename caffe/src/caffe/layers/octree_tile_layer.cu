#include "caffe/layers/octree_tile_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

#include <vector>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

namespace caffe {

__global__ void GeneMaskKernel(int* mask1_data, const int* mask0_data,
    const int* children, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int t = children[i];
    if (t == -1) continue;
    int t8 = t << 3;
    int mask0i = mask0_data[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      mask1_data[t8 + j] = mask0i;
    }
  }
}

template <typename Dtype>
__global__ void OctreeTileForward(Dtype* top_data, const int channel,
    const int top_h, const Dtype* bottom_data, const int bottom_h,
    const int* mask_data,  const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int c = i / top_h;
    int h = i % top_h;

    top_data[i] = bottom_data[c * bottom_h + mask_data[h]];
  }
}

template <typename Dtype>
__global__ void OctreeTileBackward(Dtype* bottom_data, const int channel,
    const int bottom_h, const Dtype* top_data, const int top_h,
    const int* mask_data, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int c = i / top_h;
    int h = i % top_h;
    caffe_gpu_atomic_add(top_data[i], bottom_data + c * bottom_h + mask_data[h]);
  }
}

template <typename Dtype>
void OctreeTileLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // generate the copy mask
  int nnum = octree_batch_.info().node_num(curr_depth_);
  mask0_->Reshape(vector<int> {nnum});
  int* mask0_data = mask0_->mutable_gpu_data();
  thrust::sequence(thrust::device, mask0_data, mask0_data + nnum);

  for (int d = curr_depth_; d < tile_depth_; ++d) {
    const int* children = octree_batch_.children_gpu(d);
    const int* mask0_data = mask0_->gpu_data();
    int nnum = octree_batch_.info().node_num(d);
    mask1_->Reshape(vector<int> { octree_batch_.info().node_num(d + 1) });
    int* mask1_data = mask1_->mutable_gpu_data();
    GeneMaskKernel <<< CAFFE_GET_BLOCKS(nnum), CAFFE_CUDA_NUM_THREADS>>> (
        mask1_data, mask0_data, children, nnum);
    CUDA_POST_KERNEL_CHECK;
    mask0_.swap(mask1_);
  }

  // copy according to mask0_
  int top_h = top[0]->shape(2);
  int channel = top[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int count = top_h * channel;
  const int* mask_data = mask0_->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  OctreeTileForward<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      top_data, channel, top_h, bottom_data, bottom_h, mask_data, count);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void OctreeTileLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int top_h = top[0]->shape(2);
  int channel = top[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int count = top_h * channel;
  const int* mask_data = mask0_->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* btm_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), btm_diff);
  OctreeTileBackward<Dtype> <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      btm_diff, channel, bottom_h, top_diff, top_h, mask_data, count);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeTileLayer);

}  // namespace caffe
