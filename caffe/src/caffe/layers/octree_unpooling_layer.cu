#include "caffe/layers/octree_unpooling_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void octree_unpool_forward(Dtype* top_data, const int top_h,
    const Dtype* btm_data, const int btm_h, const int* mask, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int c = i / btm_h;
    top_data[c*top_h + mask[i]] = btm_data[i];
  }
}

template <typename Dtype>
__global__ void octree_unpool_backward(Dtype* btm_diff, const int btm_h,
    const Dtype* top_diff, const int top_h, const int* mask, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int c = i / btm_h;
    btm_diff[i] = top_diff[c*top_h + mask[i]];
  }
}

template <typename Dtype>
void OctreeUnpoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // de-padding
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int buffer_h = octree_batch_.info().node_num_nempty(curr_depth_);
  octree::pad_backward_gpu<Dtype>(buffer_->mutable_gpu_data(),
      buffer_h, channel, bottom[0]->gpu_data(), bottom_h,
      octree_batch_.children_gpu(curr_depth_));

  // unpooling
  const int* mask = (const int*)bottom[1]->gpu_data();
  const Dtype* btm_data = buffer_->gpu_data();
  int top_h = top[0]->shape(2);
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = buffer_h * channel;
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  octree_unpool_forward<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>>(
      top_data, top_h, btm_data, buffer_h, mask, num);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void OctreeUnpoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int top_h = top[0]->shape(2);
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int buffer_h = octree_batch_.info().node_num_nempty(curr_depth_);
  Dtype* buffer_diff = buffer_->mutable_gpu_data();
  const int* mask = (const int*)bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  int num = buffer_h * channel;
  octree_unpool_backward<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>>(
      buffer_diff, buffer_h, top_diff, top_h, mask, num);

  octree::pad_forward_gpu<Dtype>(bottom[0]->mutable_gpu_diff(),
      bottom_h, channel, buffer_->gpu_data(),
      buffer_h, octree_batch_.children_gpu(curr_depth_));

}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeUnpoolingLayer);
}