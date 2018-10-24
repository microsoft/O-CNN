#include "caffe/layers/octree_intersection_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>

namespace caffe {

template <typename Dtype>
__global__ void copy_forward_kernel(Dtype* top, const int Htop, const int Ctop,
    const Dtype* bottom, const int Hbtm, const int* stencil, const int thread_num) {
  CUDA_KERNEL_LOOP(i, thread_num) {
    int h = i % Htop;
    int c = i / Htop;
    top[i] = bottom[c * Hbtm + stencil[h]];
  }
}

template <typename Dtype>
__global__ void copy_backward_kernel(Dtype* bottom, const int Hbtm, const int Ctop,
    const Dtype* top, const int Htop, const int* stencil, const int thread_num) {
  CUDA_KERNEL_LOOP(i, thread_num) {
    int h = i % Htop;
    int c = i / Htop;
    bottom[c * Hbtm + stencil[h]] = top[i];
  }
}

__global__ void validate_bsearch_kernel(int* idx, const unsigned* arr1, const int n1,
    const unsigned* arr2, const int n2) {
  CUDA_KERNEL_LOOP(i, n1) {
    int j = idx[i];
    if (j >= n2 || arr2[j] != arr1[i]) idx[i] = -1;
  }
}

template <typename Dtype>
void OctreeIntersectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /// calc shuffled key
  // TODO: optimize octree storage to remove the usage of shuffled_key_kernel
  int num = bottom[0]->count();
  shuffled_key_.Reshape(vector<int> { num });
  unsigned int* skey_ptr = shuffled_key_.mutable_gpu_data();
  const unsigned int* key_ptr = reinterpret_cast<const unsigned int*>(bottom[0]->gpu_data());
  octree::xyz2key_gpu(skey_ptr, key_ptr, num, curr_depth_);

  int num_gt = bottom[2]->count();
  shuffled_key_gt_.Reshape(vector<int> { num_gt });
  unsigned* skey_gt_ptr = shuffled_key_gt_.mutable_gpu_data();
  const unsigned int* key_gt_ptr = reinterpret_cast<const unsigned int*>(bottom[2]->gpu_data());
  octree::xyz2key_gpu(skey_gt_ptr, key_gt_ptr, num_gt, curr_depth_);

  /// intersection
  //// version 2.0
  index_gt_.Reshape(vector<int> {num});
  int* index_gt_ptr = index_gt_.mutable_gpu_data();
  thrust::lower_bound(thrust::device, skey_gt_ptr, skey_gt_ptr + num_gt,
      skey_ptr, skey_ptr + num, index_gt_ptr);
  validate_bsearch_kernel <<< CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>> (
      index_gt_ptr, skey_ptr, num, skey_gt_ptr, num_gt);
  index_.Reshape(vector<int> {num});
  int* index_ptr = index_.mutable_gpu_data();
  thrust::sequence(thrust::device, index_ptr, index_ptr + num);
  int* ptr1 = thrust::remove_if(thrust::device, index_ptr, index_ptr + num,
          index_gt_ptr, thrust::placeholders::_1 < 0);
  thrust::remove_if(thrust::device, index_gt_ptr, index_gt_ptr + num,
      thrust::placeholders::_1 < 0);
  int num_intersect = ptr1 - index_ptr;

  //// version 1.0
  //int num_max = std::max(num_gt, num);
  //int num_min = std::min(num_gt, num);
  //shape[0] = num_max;
  //index_all_.Reshape(shape);
  //shape[0] = num_min;
  //key_intersection_.Reshape(shape);
  //index_gt_.Reshape(shape);
  //index_.Reshape(shape);
  //int* index_all_ptr = index_all_.mutable_gpu_data();
  //thrust::sequence(thrust::device, index_all_ptr, index_all_ptr + num_max);
  //const unsigned* shuffled_key_gt_ptr = shuffled_key_gt_.gpu_data();
  //const unsigned* shuffled_key_ptr = shuffled_key_.gpu_data();
  //int* index_ptr = index_.mutable_gpu_data();
  //int* index_gt_ptr = index_gt_.mutable_gpu_data();
  //unsigned* key_intersect_ptr = key_intersection_.mutable_gpu_data();
  //thrust::pair<unsigned*, int*> end_gt = thrust::set_intersection_by_key(
  //	thrust::device, shuffled_key_gt_ptr, shuffled_key_gt_ptr + num_gt, shuffled_key_ptr,
  //	shuffled_key_ptr + num, index_all_ptr, key_intersect_ptr, index_gt_ptr);
  //thrust::pair<unsigned*, int*> end = thrust::set_intersection_by_key(
  //	thrust::device, shuffled_key_ptr, shuffled_key_ptr + num, shuffled_key_gt_ptr,
  //	shuffled_key_gt_ptr + num_gt, index_all_ptr, key_intersect_ptr, index_ptr);
  //int num_intersection_gt = end_gt.second - index_gt_ptr;
  //int num_intersect = end.second - index_ptr;
  //CHECK(num_intersect == num_intersection_gt);

  /// output data
  CHECK_EQ(bottom[1]->shape(1), bottom[3]->shape(1))
      << "Error in " << this->layer_param_.name()
          << ": The channel of the two data blobs must be the same!";
  vector<int> top_shape = bottom[1]->shape();
  top_shape[2] = num_intersect == 0 ? 1 : num_intersect;
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);
  if (num_intersect == 0) {
    // the degenarated case
    caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
    caffe_gpu_set(top[1]->count(), Dtype(0), top[1]->mutable_gpu_data());
    LOG(INFO) << "Warning: num_intersect == 0 in layer "
        << this->layer_param_.name();
  } else {
    int thread_num = top[1]->count();
    copy_forward_kernel<Dtype> <<< CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS >>> (
        top[0]->mutable_gpu_data(), top_shape[2], top_shape[1], bottom[1]->gpu_data(),
        bottom[1]->shape(2), index_.gpu_data(), thread_num);
    copy_forward_kernel<Dtype> <<< CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS >>> (
        top[1]->mutable_gpu_data(), top_shape[2], top_shape[1], bottom[3]->gpu_data(),
        bottom[3]->shape(2), index_gt_.gpu_data(), thread_num);
  }
}

template <typename Dtype>
void OctreeIntersectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[1]) { return; }

  /// gradient w.r.t  bottom[1]
  Dtype* bottom_diff1 = bottom[1]->mutable_gpu_diff();
  vector<int> shape1 = bottom[1]->shape();
  // set zero
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom_diff1);
  // copy backward
  int thread_num = top[0]->count();
  copy_backward_kernel<Dtype> <<< CAFFE_GET_BLOCKS(thread_num), CAFFE_CUDA_NUM_THREADS >>> (
      bottom_diff1, shape1[2], shape1[1], top[0]->mutable_gpu_diff(), top[0]->shape(2),
      index_.gpu_data(), thread_num);

  /// gradient w.r.t  bottom[3]
  //Dtype* bottom_diff3 = bottom[3]->mutable_gpu_diff();
  //vector<int> shape3 = bottom[3]->shape();
  //caffe_gpu_set(bottom[3]->count(), Dtype(0), bottom_diff3);
  //int thread_num = top[1]->count();
  //copy_backward_kernel<Dtype> <<< CAFFE_GET_BLOCKS(thread_num),
  //	CAFFE_CUDA_NUM_THREADS >>> (bottom_diff3, shape3[2], shape3[1],
  //		top[1]->mutable_gpu_diff(), top[1]->shape(2),
  //		index_gt_.gpu_data(), thread_num);
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeIntersectionLayer);
}