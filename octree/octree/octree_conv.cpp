#include "octree_conv.h"
#include "octree_util.h"
#include "logs.h"
#include <algorithm>

namespace octree {

template <typename Dtype>
void OctreeBaseConv<Dtype>::setup(const vector<int>& kernel_size, const int stride,
    const int curr_depth, const int channel_in, const int channel_out) {
  // kernel size
  kernel_size_ = kernel_size;
  CHECK(kernel_size_[0] < 4 && kernel_size_[1] < 4 && kernel_size_[2] < 4)
      << "kernel_size should be less than 4";

  // stride
  stride_ = stride;
  CHECK(stride_ <= 2) << "stride should be less than 2";

  // special case: im2col is the identity for 1x1 convolution with stride 1
  is_1x1_ = kernel_size_[0] == 1 && kernel_size_[1] == 1 &&
            kernel_size_[2] == 1 && stride_ == 1;

  // current octree depth
  curr_depth_ = curr_depth;

  // channels & num_output_
  channels_ = conv_in_channels_ = channel_in;
  num_output_ = conv_out_channels_ = channel_out;
  if (is_deconvolution_layer()) {
    std::swap(conv_out_channels_, conv_in_channels_);
  }

  kernel_sdim_ = kernel_size_[0] * kernel_size_[1] * kernel_size_[2];
  kernel_dim_ = kernel_sdim_ * conv_in_channels_;

  ni_cpu_ptr_ = NeighHelper::get_ni(kernel_size_).data();
  ni_gpu_ptr_ = nullptr; // must be set before using
}


template <typename Dtype>
void OctreeBaseConv<Dtype>::reshape() {
  // weight shape
  weights_shape_ = vector<int> {conv_out_channels_, conv_in_channels_ * kernel_sdim_};

  // compute top shape
  int btm_h = octree_.info().node_num(curr_depth_);
  int top_blob_depth = curr_depth_, top_h = btm_h;
  if (stride_ == 2) {
    if (is_deconvolution_layer()) {
      top_blob_depth++;
      top_h = octree_.info().node_num(top_blob_depth);
    } else {
      top_blob_depth--;
      top_h = octree_.info().node_num_nempty(top_blob_depth);
    }
    CHECK(0 <= top_blob_depth && top_blob_depth <= octree_.info().depth());
  }
  if (top_h == 0) top_h = 1;      // avoid degenerated case
  top_shape_ = vector<int> { 1, num_output_, top_h, 1 };

  // reshape workspce
  workspace_depth_ = curr_depth_; // the depth value used for octree2col
  if (is_deconvolution_layer() && stride_ == 2) workspace_depth_++;
  workspace_h_ = btm_h;
  if (stride_ == 2) {
    if (is_deconvolution_layer()) { workspace_h_ = top_h >> 3; }
    else { workspace_h_ = btm_h >> 3; }
  }
  workspace_n_ = 1;
  workspace_ha_ = workspace_h_;
  int ideal_size = workspace_h_ * kernel_dim_;
  if (ideal_size > MAX_SIZE && !is_1x1_) {
    workspace_n_ = (ideal_size + MAX_SIZE - 1) / MAX_SIZE;
    workspace_ha_ = (workspace_h_ + workspace_n_ - 1) / workspace_n_;
  }
  workspace_shape_ = vector<int> { kernel_dim_, workspace_ha_};

  // reshape result_buffer_
  if (workspace_n_ > 1) {
    result_buffer_shape_ = vector<int> { conv_out_channels_, workspace_ha_ };
  } else {
    result_buffer_shape_.clear();
  }

  // reshape data_buffer_
  if (stride_ == 2) {
    data_buffer_shape_ = vector<int> { 1, conv_out_channels_, workspace_h_, 1 };
  } else {
    data_buffer_shape_.clear();
  }
}


template <typename Dtype>
void OctreeBaseConv<Dtype>::forward_cpu_gemm(Dtype* top_data,
    const Dtype* bottom_data, const Dtype* weights) {
  const Dtype* col_data = bottom_data;
  Dtype* result_data = workspace_n_ == 1 ? top_data : result_buffer_;
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree2col_cpu<Dtype>(workspace_,
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_.neighbor_cpu(workspace_depth_),
          ni_cpu_ptr_, workspace_ha_, n);
      col_data = workspace_;
    }

    engine_cpu_->gemm(false, false, conv_out_channels_,
        workspace_ha_, kernel_dim_, Dtype(1.0), weights, col_data,
        Dtype(0), result_data);

    if (workspace_n_ == 1) return;
    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    for (int c = 0; c < conv_out_channels_; ++c) {
      memcpy_cpu(num, result_data + c * workspace_ha_,
          top_data + c * workspace_h_ + n * workspace_ha_);
    }
  }
}

template <typename Dtype>
void OctreeBaseConv<Dtype>::backward_cpu_gemm(Dtype* bottom_diff,
    const Dtype* top_diff, const Dtype* weights) {
  Dtype* col_diff = is_1x1_ ? bottom_diff : workspace_;
  for (int n = 0; n < workspace_n_; ++n) {
    const Dtype* result_buffer = top_diff;
    if (workspace_n_ > 1) {
      Dtype* buffer_ = result_buffer_;
      int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
      for (int c = 0; c < conv_out_channels_; ++c) {
        memcpy_cpu(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer_ + c * workspace_ha_);
      }
      result_buffer = result_buffer_;
    }

    engine_cpu_->gemm(true, false, kernel_dim_,
        workspace_ha_, conv_out_channels_, Dtype(1.0), weights,
        result_buffer, Dtype(0.0), col_diff);

    if (!is_1x1_) {
      col2octree_cpu<Dtype>(col_diff, bottom_diff,
          conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_.neighbor_cpu(workspace_depth_),
          ni_cpu_ptr_, workspace_ha_, n);
    }
  }
}


template <typename Dtype>
void OctreeBaseConv<Dtype>::weight_cpu_gemm(Dtype* weights_diff,
    const Dtype* bottom_data, const Dtype* top_diff) {
  int num = num_elements(weights_shape_);
  memset_cpu(num, Dtype(0), weights_diff);

  const Dtype* col_data = bottom_data;
  const Dtype* result_buffer = top_diff;
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree2col_cpu<Dtype>(workspace_,
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_.neighbor_cpu(workspace_depth_),
          ni_cpu_ptr_, workspace_ha_, n);
      col_data = workspace_;
    }

    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    if (workspace_n_ > 1) {
      Dtype* buffer = result_buffer_;
      for (int c = 0; c < conv_out_channels_; ++c) {
        memcpy_cpu(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer + c * workspace_ha_);
      }
      result_buffer = result_buffer_;
    }

    engine_cpu_->gemm(false, true, conv_out_channels_,
        kernel_dim_, workspace_ha_, Dtype(1.0), result_buffer, col_data,
        Dtype(1.0), weights_diff);
  }
}

#ifdef USE_CUDA

template <typename Dtype>
void OctreeBaseConv<Dtype>::forward_gpu_gemm(Dtype* top_data,
    const Dtype* bottom_data, const Dtype* weights) {
  const Dtype* col_data = bottom_data;
  Dtype* result_data = workspace_n_ == 1 ? top_data : result_buffer_;
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree2col_gpu<Dtype>(workspace_,
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_.neighbor_gpu(workspace_depth_),
          ni_gpu_ptr_, workspace_ha_, n);
      col_data = workspace_;
    }

    engine_gpu_->gemm(false, false, conv_out_channels_,
        workspace_ha_, kernel_dim_, Dtype(1.0), weights, col_data,
        Dtype(0), result_data);

    if (workspace_n_ == 1) return;
    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    for (int c = 0; c < conv_out_channels_; ++c) {
      memcpy_gpu(num, result_data + c * workspace_ha_,
          top_data + c * workspace_h_ + n * workspace_ha_);
    }
  }
}

template <typename Dtype>
void OctreeBaseConv<Dtype>::backward_gpu_gemm(Dtype* bottom_diff,
    const Dtype* top_diff, const Dtype* weights) {
  Dtype* col_diff = is_1x1_ ? bottom_diff : workspace_;
  for (int n = 0; n < workspace_n_; ++n) {
    const Dtype* result_buffer = top_diff;
    if (workspace_n_ > 1) {
      Dtype* buffer_ = result_buffer_;
      int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
      for (int c = 0; c < conv_out_channels_; ++c) {
        memcpy_gpu(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer_ + c * workspace_ha_);
      }
      result_buffer = result_buffer_;
    }

    engine_gpu_->gemm(true, false, kernel_dim_,
        workspace_ha_, conv_out_channels_, Dtype(1.0), weights,
        result_buffer, Dtype(0.0), col_diff);

    if (!is_1x1_) {
      col2octree_gpu<Dtype>(col_diff, bottom_diff,
          conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_.neighbor_gpu(workspace_depth_),
          ni_gpu_ptr_, workspace_ha_, n);
    }
  }
}


template <typename Dtype>
void OctreeBaseConv<Dtype>::weight_gpu_gemm(Dtype* weights_diff,
    const Dtype* bottom_data, const Dtype* top_diff) {
  int num = num_elements(weights_shape_);
  memset_gpu(num, Dtype(0), weights_diff);

  const Dtype* col_data = bottom_data;
  const Dtype* result_buffer = top_diff;
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree2col_gpu<Dtype>(workspace_,
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_.neighbor_gpu(workspace_depth_),
          ni_gpu_ptr_, workspace_ha_, n);
      col_data = workspace_;
    }

    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    if (workspace_n_ > 1) {
      Dtype* buffer = result_buffer_;
      for (int c = 0; c < conv_out_channels_; ++c) {
        memcpy_gpu(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer + c * workspace_ha_);
      }
      result_buffer = result_buffer_;
    }

    engine_gpu_->gemm(false, true, conv_out_channels_,
        kernel_dim_, workspace_ha_, Dtype(1.0), result_buffer, col_data,
        Dtype(1.0), weights_diff);
  }
}

#endif // USE_CUDA

template class OctreeBaseConv<float>;
template class OctreeBaseConv<double>;

}  // namespace octree
