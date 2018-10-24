#include "caffe/layers/octree_zeroize_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void zeroize_kernel(Dtype* des, const Dtype* src,
    const Dtype* label_data, const int height, const int mask, const int n) {
  CUDA_KERNEL_LOOP(i, n) {
    int h = i % height;
    des[i] = static_cast<int>(label_data[h]) == mask ? Dtype(0) : src[i];
  }
}

template <typename Dtype>
void OctreeZeroizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int blob_num = top.size();
  int height = bottom[blob_num]->count();
  const Dtype* label_data = bottom[blob_num]->gpu_data();
  for (int i = 0; i < blob_num; ++i) {
    int channel = bottom[i]->shape(1);
    int num = channel * height;
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    zeroize_kernel<Dtype> <<< CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>> (
        top_data, bottom_data, label_data, height, mask_, num);
    CUDA_POST_KERNEL_CHECK;
  }
}


template <typename Dtype>
void OctreeZeroizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int blob_num = top.size();
  int height = bottom[blob_num]->count();
  const Dtype* label_data = bottom[blob_num]->gpu_data();
  for (int i = 0; i < blob_num; ++i) {
    if (!propagate_down[i]) continue;
    int channel = bottom[i]->shape(1);
    int num = channel * height;
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    const Dtype* top_diff = top[i]->gpu_diff();
    zeroize_kernel<Dtype> <<< CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >>> (
        bottom_diff, top_diff, label_data, height, mask_, num);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OctreeZeroizeLayer);

}  // namespace caffe