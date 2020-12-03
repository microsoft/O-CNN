#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/octree_base_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
OctreeBaseConvLayer<Dtype>::~OctreeBaseConvLayer() {
#if defined(USE_CUDNN) && !defined(CPU_ONLY)
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
  cudnnDestroyFilterDescriptor(filter_desc_);
  cudnnDestroy(handle_);
#endif // USE_CUDNN
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // kernel size
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  size_t kss = conv_param.kernel_size_size();
  CHECK(kss == 1 || kss == 3) << "Specify 1 or 3 kernel_size paramters";
  kernel_size_.resize(3);
  if (kss == 1) {
    kernel_size_[0] = conv_param.kernel_size(0);
    kernel_size_[1] = conv_param.kernel_size(0);
    kernel_size_[2] = conv_param.kernel_size(0);
  } else {
    kernel_size_[0] = conv_param.kernel_size(0);
    kernel_size_[1] = conv_param.kernel_size(1);
    kernel_size_[2] = conv_param.kernel_size(2);
  }
  CHECK(kernel_size_[0] < 4 && kernel_size_[1] < 4 && kernel_size_[2] < 4)
      << "kernel_size should be less than 4";
  kernel_sdim_ = kernel_size_[0] * kernel_size_[1] * kernel_size_[2];

  // stride
  CHECK_LE(conv_param.stride_size(), 1);
  stride_ = conv_param.stride_size() == 0 ? 1 : conv_param.stride(0);
  CHECK_LE(stride_, 2) << "stride should be less than 2";

  // special case: im2col is the identity for 1x1 convolution with stride 1
  is_1x1_ = kernel_size_[0] == 1 && kernel_size_[1] == 1 &&
      kernel_size_[2] == 1 && stride_ == 1;

  // current octree depth
  CHECK(this->layer_param_.octree_param().has_curr_depth())
      << "Error in " << this->layer_param_.name() << ": "
      << "The octree depth of bottom blob should be set coreectly.";
  curr_depth_ = this->layer_param_.octree_param().curr_depth();
  //curr_depth_ = Octree::get_curr_depth();
  //if (stride_ == 2) {
  //	if (is_deconvolution_layer()) Octree::set_curr_depth(curr_depth_ + 1);
  //	else Octree::set_curr_depth(curr_depth_ - 1);
  //}

  // channels & num_output_
  channels_ = conv_in_channels_ = bottom[0]->shape(1);
  num_output_ = conv_out_channels_ = conv_param.num_output();
  if (is_deconvolution_layer()) {
    std::swap(conv_out_channels_, conv_in_channels_);
  }

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape{ conv_out_channels_, conv_in_channels_, kernel_sdim_, 1 };
  vector<int> bias_shape;
  bias_term_ = conv_param.bias_term();
  if (bias_term_) bias_shape.push_back(num_output_);

  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string()
          << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string()
          << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) this->blobs_.resize(2);
    else this->blobs_.resize(1);

    // Initialize and fill the weights:
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype>> weight_filler(
            GetFiller<Dtype>(conv_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(conv_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // set the pointer of workspace
  workspace_ = Octree::get_workspace(Dtype(0), 0);
  result_buffer_ = Octree::get_workspace(Dtype(0), 1);
  // for the deconv layer, the following code is wrong in the backward pass
  //data_buffer_ = Octree::get_workspace(Dtype(0), 2);
  data_buffer_.reset(new Blob<Dtype>());

#if defined(USE_CUDNN) && !defined(CPU_ONLY)
  // cudnn
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createFilterDesc<Dtype>(&filter_desc_, weight_shape[0],
      weight_shape[1], weight_shape[2], weight_shape[3]);
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createConvolutionDesc<Dtype>(&conv_desc_);
  filter_workspace_size_ = 8 * 1024 * 1024;
  vector<int> filter_workspace_shape(1, filter_workspace_size_);
  filter_workspace_.Reshape(filter_workspace_shape);
#endif //USE_CUDNN
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // a workaround for the first time reshape
  if (top[0]->count() == 0) {
    vector<int> top_shape = bottom[0]->shape();
    top_shape[1] = num_output_;
    top_shape[2] = 8;
    top[0]->Reshape(top_shape);
    return;
  }

  // set octree_batch_
  bool octree_in = bottom.size() == 2;
  Blob<Dtype>& the_octree = octree_in ? *bottom[1] : Octree::get_octree(Dtype(0));
  octree::set_octree_parser(octree_batch_, the_octree);

  // check bottom shape
  int bottom_h = bottom[0]->shape(2);
  int nnum = octree_batch_.info().node_num(curr_depth_);
  if (nnum != 0) {
    CHECK_EQ(bottom_h, nnum)
        << "The octree node number is inconsistent in " << this->layer_param_.name()
            << ", and the curr_depth_ = " << curr_depth_;
  } else {
    CHECK_EQ(bottom_h, 1);
    LOG(INFO) << "Warning in layer" << this->layer_param().name()
        << ": This octree layer is empty";
  }

  // compute top shape
  int top_blob_depth = curr_depth_;
  if (stride_ == 2) {
    if (is_deconvolution_layer()) top_blob_depth++;
    else top_blob_depth--;
    CHECK(0 <= top_blob_depth && top_blob_depth <= octree_batch_.info().depth())
        << "Error in " << this->layer_param_.name() << ": top_blob_depth "
            << top_blob_depth << "- octree_depth " << octree_batch_.info().depth();
  }
  int top_h = octree_batch_.info().node_num(top_blob_depth);
  if (top_h == 0) top_h = 1;
  vector<int> top_shape{ 1, num_output_, top_h, 1 };
  top[0]->Reshape(top_shape);

  // reshape workspce
  workspace_depth_ = curr_depth_;
  if (is_deconvolution_layer() && stride_ == 2) workspace_depth_++;
  workspace_h_ = bottom_h;
  if (stride_ == 2) {
    if (is_deconvolution_layer()) workspace_h_ = top_h >> 3;
    else workspace_h_ = bottom_h >> 3;
  }
  workspace_n_ = 1;
  workspace_ha_ = workspace_h_;
  // In some other layers, such as octree_pooling, the workspace blob is also
  // resized, without considering the get_workspace_maxsize(), the actual
  // size of workspace might be larger than get_workspace_maxsize(), so 
  // calculate the MAX_SIZE from get_workspace_maxsize() and its original count()
  const int MAX_SIZE = std::max(Octree::get_workspace_maxsize(), workspace_->count());
  int ideal_size = workspace_h_ * kernel_dim_;
  if (ideal_size > MAX_SIZE && !is_1x1_) {
    workspace_n_ = (ideal_size + MAX_SIZE - 1) / MAX_SIZE;
    workspace_ha_ = (workspace_h_ + workspace_n_ - 1) / workspace_n_;
  }
  workspace_->Reshape(vector<int> { 1, kernel_dim_, workspace_ha_, 1 });

  // reshape result_buffer_
  if (workspace_n_ > 1) {
    result_buffer_->Reshape(vector<int> { conv_out_channels_, workspace_ha_ });
  }

  // reshape data_buffer_
  if (stride_ == 2) {
    data_buffer_shape_ = vector<int> { 1, conv_out_channels_, workspace_h_, 1 };
    data_buffer_->Reshape(data_buffer_shape_);
  }

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_h_ = is_deconvolution_layer() ? top_h : workspace_h_;
    bias_multiplier_.Reshape(vector<int> { bias_multiplier_h_ });
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::forward_cpu_gemm(Dtype* top_data,
    const Dtype* bottom_data, const Dtype* weights) {
  const Dtype* col_data = bottom_data;
  Dtype* result_data = workspace_n_ == 1 ? top_data :
      result_buffer_->mutable_cpu_data();
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree::octree2col_cpu<Dtype>(workspace_->mutable_cpu_data(),
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_batch_.neighbor_cpu(workspace_depth_),
          Octree::get_ni(kernel_size_)->cpu_data(), workspace_ha_, n);
      col_data = workspace_->cpu_data();
    }

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
        workspace_ha_, kernel_dim_, Dtype(1.0), weights, col_data,
        Dtype(0), result_data);

    if (workspace_n_ == 1) return;
    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    for (int c = 0; c < conv_out_channels_; ++c) {
      caffe_copy(num, result_data + c * workspace_ha_,
          top_data + c * workspace_h_ + n * workspace_ha_);
    }
  }
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::forward_cpu_bias(Dtype* top_data,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      bias_multiplier_h_, 1, Dtype(1.0), bias, bias_multiplier_.cpu_data(),
      Dtype(1.0), top_data);
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::backward_cpu_gemm(Dtype* bottom_diff,
    const Dtype* top_diff, const Dtype* weights) {
  Dtype* col_diff = is_1x1_ ? bottom_diff : workspace_->mutable_cpu_data();
  for (int n = 0; n < workspace_n_; ++n) {
    const Dtype* result_buffer = top_diff;
    if (workspace_n_ > 1) {
      Dtype* buffer_ = result_buffer_->mutable_cpu_data();
      int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
      for (int c = 0; c < conv_out_channels_; ++c) {
        caffe_copy(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer_ + c * workspace_ha_);
      }
      result_buffer = result_buffer_->cpu_data();
    }

    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        workspace_ha_, conv_out_channels_, Dtype(1.0), weights,
        result_buffer, Dtype(0.0), col_diff);

    if (!is_1x1_) {
      octree::col2octree_cpu<Dtype>(col_diff, bottom_diff,
          conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_batch_.neighbor_cpu(workspace_depth_),
          Octree::get_ni(kernel_size_)->cpu_data(), workspace_ha_, n);
    }
  }
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::weight_cpu_gemm(Dtype* weights_diff,
    const Dtype* bottom_data, const Dtype* top_diff) {
  const Dtype* col_data = bottom_data;
  const Dtype* result_buffer = top_diff;
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree::octree2col_cpu<Dtype>(workspace_->mutable_cpu_data(),
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_batch_.neighbor_cpu(workspace_depth_),
          Octree::get_ni(kernel_size_)->cpu_data(), workspace_ha_, n);
      col_data = workspace_->cpu_data();
    }

    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    if (workspace_n_ > 1) {
      Dtype* buffer = result_buffer_->mutable_cpu_data();
      for (int c = 0; c < conv_out_channels_; ++c) {
        caffe_copy(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer + c * workspace_ha_);
      }
      result_buffer = result_buffer_->cpu_data();
    }

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_,
        kernel_dim_, workspace_ha_, Dtype(1.0), result_buffer, col_data,
        Dtype(1.0), weights_diff);
  }
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::backward_cpu_bias(Dtype* bias_diff,
    const Dtype* top_diff) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, bias_multiplier_h_,
      Dtype(1.0), top_diff, bias_multiplier_.cpu_data(), 1., bias_diff);
}

#ifndef CPU_ONLY
template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::forward_gpu_gemm(Dtype* top_data,
    const Dtype* bottom_data, const Dtype* weights) {
  const Dtype* col_data = bottom_data;
  Dtype* result_data = workspace_n_ == 1 ? top_data :
      result_buffer_->mutable_gpu_data();
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree::octree2col_gpu<Dtype>(workspace_->mutable_gpu_data(),
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_batch_.neighbor_gpu(workspace_depth_),
          Octree::get_ni(kernel_size_)->gpu_data(), workspace_ha_, n);
      col_data = workspace_->gpu_data();
    }

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
        workspace_ha_, kernel_dim_, Dtype(1.0), weights, col_data,
        Dtype(0), result_data);

    if (workspace_n_ == 1) return;
    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    for (int c = 0; c < conv_out_channels_; ++c) {
      caffe_copy(num, result_data + c * workspace_ha_,
          top_data + c * workspace_h_ + n * workspace_ha_);
    }
  }
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::forward_gpu_bias(Dtype* top_data,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      bias_multiplier_h_, 1, Dtype(1.0), bias, bias_multiplier_.gpu_data(),
      Dtype(1.0), top_data);
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::backward_gpu_gemm(Dtype* bottom_diff,
    const Dtype* top_diff, const Dtype* weights) {
  Dtype* col_diff = is_1x1_ ? bottom_diff : workspace_->mutable_gpu_data();
  for (int n = 0; n < workspace_n_; ++n) {
    const Dtype* result_buffer = top_diff;
    if (workspace_n_ > 1) {
      Dtype* buffer_ = result_buffer_->mutable_gpu_data();
      int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
      for (int c = 0; c < conv_out_channels_; ++c) {
        caffe_copy(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer_ + c * workspace_ha_);
      }
      result_buffer = result_buffer_->gpu_data();
    }

    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        workspace_ha_, conv_out_channels_, Dtype(1.0), weights,
        result_buffer, Dtype(0.0), col_diff);

    if (!is_1x1_) {
      octree::col2octree_gpu<Dtype>(col_diff, bottom_diff,
          conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_batch_.neighbor_gpu(workspace_depth_),
          Octree::get_ni(kernel_size_)->gpu_data(), workspace_ha_, n);
    }
  }
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::weight_gpu_gemm(Dtype* weights_diff,
    const Dtype* bottom_data, const Dtype* top_diff) {
  const Dtype* col_data = bottom_data;
  const Dtype* result_buffer = top_diff;
  for (int n = 0; n < workspace_n_; ++n) {
    if (!is_1x1_) {
      octree::octree2col_gpu<Dtype>(workspace_->mutable_gpu_data(),
          bottom_data, conv_in_channels_, workspace_h_, kernel_sdim_,
          stride_, octree_batch_.neighbor_gpu(workspace_depth_),
          Octree::get_ni(kernel_size_)->gpu_data(), workspace_ha_, n);
      col_data = workspace_->gpu_data();
    }

    int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
    if (workspace_n_ > 1) {
      Dtype* buffer = result_buffer_->mutable_gpu_data();
      for (int c = 0; c < conv_out_channels_; ++c) {
        caffe_copy(num, top_diff + c * workspace_h_ + n * workspace_ha_,
            buffer + c * workspace_ha_);
      }
      result_buffer = result_buffer_->gpu_data();
    }

#if defined(USE_CUDNN) && !defined(CPU_ONLY)
    if (n == 0) {
      // Interestingly, we find the GEMM is nearly ten times slower than Cudnn
      // when doing weight back-propagation. It's possible because the ill matrix
      // size (such as A(1000, 120000) * B(120000, 500) = W(1000, 500)).
      cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, 1, conv_in_channels_,
          kernel_dim_ / conv_in_channels_, workspace_ha_);
      cudnn::setTensor4dDesc<Dtype>(&top_desc_, 1, conv_out_channels_,
          1, workspace_ha_);
      cudnn::setConvolutionDesc<Dtype>(&conv_desc_, bottom_desc_, filter_desc_,
          0, 0, 1, 1);
      // choose backward algorithm for filter
      bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
      CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
              bottom_desc_, top_desc_, conv_desc_, filter_desc_,
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              filter_workspace_size_, &bwd_filter_algo_));
    }
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle_,
            cudnn::dataType<Dtype>::one, bottom_desc_, col_data, top_desc_,
            result_buffer, conv_desc_, bwd_filter_algo_,
            filter_workspace_.mutable_gpu_data(), filter_workspace_size_,
            cudnn::dataType<Dtype>::one, filter_desc_, weights_diff));
#else // GEMM			
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_,
        kernel_dim_, workspace_ha_, Dtype(1.0), result_buffer, col_data,
        Dtype(1.0), weights_diff);
#endif
  }
}

template <typename Dtype>
void OctreeBaseConvLayer<Dtype>::backward_gpu_bias(Dtype* bias_diff,
    const Dtype* top_diff) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, bias_multiplier_h_,
      Dtype(1.0), top_diff, bias_multiplier_.gpu_data(), 1., bias_diff);
}

#endif

INSTANTIATE_CLASS(OctreeBaseConvLayer);

}  // namespace caffe
