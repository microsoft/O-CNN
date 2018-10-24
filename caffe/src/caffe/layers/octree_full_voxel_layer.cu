#include "caffe/layers/octree_full_voxel_layer.hpp"
#include "caffe/util/gpu_util.cuh"
namespace caffe {

template <typename Dtype>
__global__ void octree2voxel_forward(Dtype* top_data,
    const Dtype* bottom_data, const int channel, const int bottom_h,
    const unsigned* xyz2key, const int voxel_num, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int k = i % voxel_num;
    int t = i / voxel_num;
    int c = t % channel;
    int n = t / channel;
    top_data[i] = bottom_data[c * bottom_h + n * voxel_num + xyz2key[k]];
  }
}

template <typename Dtype>
__global__ void octree2voxel_backward(const Dtype* top_diff,
    Dtype* bottom_diff, const int channel, const int bottom_h,
    const unsigned* xyz2key, const int voxel_num, const int nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int k = i % voxel_num;
    int t = i / voxel_num;
    int c = t % channel;
    int n = t / channel;
    bottom_diff[c * bottom_h + n * voxel_num + xyz2key[k]] = top_diff[i];
  }
}

template <typename Dtype>
void Octree2FullVoxelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // calc mapping
  if (index_mapper_.count() == 0) build_mapping(curr_depth_);

  int voxel_num = 1 << 3 * curr_depth_;
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int nthreads = batch_size_ * channel * voxel_num;
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const unsigned* xyz_to_key = index_mapper_.gpu_data();
  octree2voxel_forward<Dtype> <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>>(
      top_data, bottom_data, channel, bottom_h, index_mapper_.gpu_data(), voxel_num, nthreads);
}

template <typename Dtype>
void Octree2FullVoxelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  int voxel_num = 1 << 3 * curr_depth_;
  int channel = bottom[0]->shape(1);
  int bottom_h = bottom[0]->shape(2);
  int nthreads = batch_size_ * channel * voxel_num;
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const unsigned* xyz_to_key = index_mapper_.gpu_data();
  octree2voxel_backward<Dtype> <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >>>(
      top_diff, bottom_diff, channel, bottom_h, index_mapper_.gpu_data(), voxel_num, nthreads);
}

INSTANTIATE_LAYER_GPU_FUNCS(Octree2FullVoxelLayer);

}