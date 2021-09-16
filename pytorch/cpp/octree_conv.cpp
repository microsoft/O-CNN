#include <ATen/cuda/CUDAContext.h>
#include <octree/octree_conv.h>
#include <octree/octree_nn.h>

#include "ocnn.h"

namespace {

using octree::OctreeBaseConv;

// used for debug
template <typename dtype>
void dump_tensor(const Tensor tensor, string filename="") {
  int dim = tensor.dim();
  filename  +=  "_shape";
  for (int j = 0; j < dim; ++j) {
    filename += "_" + std::to_string(tensor.size(j));
  }

  std::cout << filename << std::endl;

  Tensor t = tensor.cpu();
  int n = t.numel();
  std::ofstream outfile(filename, std::ios::binary);
  const dtype* ptr = t.data_ptr<dtype>();
  outfile.write((char*) ptr, n * sizeof(dtype));
  outfile.close();
}

class THGpuGemm : public octree::GEMMEngine<float> {
 public:
  virtual void gemm(const bool TransA, const bool TransB, const int M,
                    const int N, const int K, const float alpha, const float* A,
                    const float* B, const float beta, float* C) override {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Note that cublas follows fortran order.
    cublasOperation_t opa = TransA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opb = TransB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int64_t lda = TransA ? M : K;
    int64_t ldb = TransB ? K : N;
    cublasStatus_t status = cublasSgemm(handle, opb, opa, N, M, K, &alpha, B,
                                        ldb, A, lda, &beta, C, N);
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "Cublas error in THGpuGemm";
  }
};

class OctreeConvTH : public OctreeBaseConv<float> {
 public:
  explicit OctreeConvTH(int depth, int num_output, vector<int> kernel_size,
                        int stride, bool nempty)
      : depth_(depth), num_output_(num_output), kernel_size_(kernel_size),
        stride_(stride), non_empty_(nempty) {
    CHECK_GT(depth_, 0) << "The depth should be larger than 0";
    CHECK_GT(num_output_, 0) << "The num_output should be larger than 0";
    for (auto k : kernel_size_) {
      CHECK(0 < k && k < 4) << "Invalide kernel size";
    }
    CHECK(stride_ == 1 || stride_ == 2) << "Unsupport stride";
  }

  void setup_op(Tensor data_in, Tensor octree_in) {
    this->engine_gpu_ = &th_gpu_gemm_;

    // setup octree
    auto octree_ptr = octree_in.data_ptr<uint8_t>();
    this->octree_.set_gpu(octree_ptr);

    // setup octree conv
    int channel_in = data_in.size(1), height_btm = data_in.size(2);
    OctreeBaseConv<float>::setup(
        kernel_size_, stride_, depth_, channel_in, num_output_, non_empty_);
    if ((stride_ == 2 && is_deconvolution_layer()) || non_empty_) {
      CHECK_EQ(height_btm, this->octree_.info().node_num_nempty(depth_))
          << ", d: " << depth_ << ", channel_in: " << channel_in;
    } else {
      CHECK_EQ(height_btm, this->octree_.info().node_num(depth_))
          << ", d: " << depth_ << ", channel_in: " << channel_in;
    }
  }

  vector<Tensor> alloc_temp_memory(torch::TensorOptions options) {
    OctreeBaseConv<float>::reshape();

    vector<Tensor> tmp_tensors;
    int count = num_elements(this->workspace_shape_);
    Tensor workspace = torch::zeros({count}, options);
    this->workspace_ = workspace.data_ptr<float>();
    tmp_tensors.push_back(workspace);

    count = num_elements(this->result_buffer_shape_);
    if (count != 0) {
      Tensor result_buffer = torch::zeros({count}, options);
      this->result_buffer_ = result_buffer.data_ptr<float>();
      tmp_tensors.push_back(result_buffer);
    } else {
      this->result_buffer_ = nullptr;
    }

    vector<int>& ni_cpu = NeighHelper::get_ni(kernel_size_);
    count = ni_cpu.size();
    if (count != 0) {
      Tensor ni_gpu = torch::zeros({count}, options.dtype(torch::kInt32));
      auto ni_ptr = ni_gpu.data_ptr<int>();
      memcpy_gpu(count, ni_cpu.data(), ni_gpu.data_ptr<int>());
      this->ni_gpu_ptr_ = ni_ptr;
      tmp_tensors.push_back(ni_gpu);
    }

    if (non_empty_) {
      this->child_ = octree_.children_gpu(this->workspace_depth_);
      Tensor t0 = torch::arange(this->child_h_, options.dtype(torch::kInt32));
      Tensor t1 = torch::zeros(this->ichild_h_, options.dtype(torch::kInt32));
      this->ichild_ = t1.data_ptr<int>();
      pad_backward_gpu(t1.data_ptr<int>(), this->ichild_h_, 1,
                       t0.data_ptr<int>(), this->child_h_, this->child_);
      tmp_tensors.push_back(t1);
    }

    return tmp_tensors;
  }

 private:
  int depth_;
  int num_output_;
  vector<int> kernel_size_;
  int stride_;
  bool non_empty_;
  THGpuGemm th_gpu_gemm_;
};

class OctreeConvOp : public OctreeConvTH {
 public:
  explicit OctreeConvOp(int depth, int num_output, vector<int> kernel_size,
                        int stride, bool nempty)
      : OctreeConvTH(depth, num_output, kernel_size, stride, nempty) {}

  Tensor compute(Tensor data_in, Tensor weights, Tensor octree) {
    // init
    this->setup_op(data_in, octree);
    torch::TensorOptions options = data_in.options();
    vector<Tensor> tmp_tensors = this->alloc_temp_memory(options);
    Tensor data_out =
        torch::zeros({1, this->top_shape_[1], this->top_shape_[2], 1}, options);

    // get pointers
    data_in = data_in.contiguous();
    const float* btm_data = data_in.data_ptr<float>();
    const float* weights_data = weights.data_ptr<float>();
    float* top_data = data_out.data_ptr<float>();

    // forward
    this->forward_gpu_gemm(top_data, btm_data, weights_data);
    return data_out;
  }

  virtual bool is_deconvolution_layer() override { return false; }
};

class OctreeConvGradOp : public OctreeConvTH {
 public:
  explicit OctreeConvGradOp(int depth, int num_output, vector<int> kernel_size,
                            int stride, bool nempty)
      : OctreeConvTH(depth, num_output, kernel_size, stride, nempty) {}

  vector<Tensor> compute(Tensor data_in, Tensor weights, Tensor octree,
                         Tensor diff_in) {
    // init
    this->setup_op(data_in, octree);
    vector<Tensor> tmp_tensors = this->alloc_temp_memory(data_in.options());
    Tensor btm_out = torch::zeros_like(data_in);
    Tensor weights_out = torch::zeros_like(weights);

    // get points
    data_in = data_in.contiguous();
    diff_in = diff_in.contiguous();
    auto btm_data = data_in.data_ptr<float>();
    auto weights_data = weights.data_ptr<float>();
    auto top_diff = diff_in.data_ptr<float>();
    auto btm_diff = btm_out.data_ptr<float>();
    auto weights_diff = weights_out.data_ptr<float>();

    // backward, TODO: add judgement here
    this->weight_gpu_gemm(weights_diff, btm_data, top_diff);
    this->backward_gpu_gemm(btm_diff, top_diff, weights_data);
    return {btm_out, weights_out};
  }

  virtual bool is_deconvolution_layer() override { return false; }
};

class OctreeDeconvOp : public OctreeConvTH {
 public:
  explicit OctreeDeconvOp(int depth, int num_output, vector<int> kernel_size,
                          int stride, bool nempty)
      : OctreeConvTH(depth, num_output, kernel_size, stride, nempty) {}

  Tensor compute(Tensor data_in, Tensor weights, Tensor octree) {
    // init
    this->setup_op(data_in, octree);
    torch::TensorOptions options = data_in.options();
    vector<Tensor> tmp_tensors = this->alloc_temp_memory(options);
    Tensor data_out =
        torch::zeros({1, this->top_shape_[1], this->top_shape_[2], 1}, options);

    // get pointers
    data_in = data_in.contiguous();
    const float* btm_data = data_in.data_ptr<float>();
    const float* weights_data = weights.data_ptr<float>();
    float* top_data = data_out.data_ptr<float>();

    // forward
    this->backward_gpu_gemm(top_data, btm_data, weights_data);
    return data_out;
  }

  virtual bool is_deconvolution_layer() override { return true; }
};

class OctreeDeconvGradOp : public OctreeConvTH {
 public:
  explicit OctreeDeconvGradOp(int depth, int num_output,
                              vector<int> kernel_size, int stride, bool nempty)
      : OctreeConvTH(depth, num_output, kernel_size, stride, nempty) {}

  vector<Tensor> compute(Tensor data_in, Tensor weights, Tensor octree,
                         Tensor diff_in) {
    // init
    this->setup_op(data_in, octree);
    vector<Tensor> tmp_tensors = this->alloc_temp_memory(data_in.options());
    Tensor btm_out = torch::zeros_like(data_in);
    Tensor weights_out = torch::zeros_like(weights);

    // get points
    data_in = data_in.contiguous();
    diff_in = diff_in.contiguous();
    auto btm_data = data_in.data_ptr<float>();
    auto weights_data = weights.data_ptr<float>();
    auto top_diff = diff_in.data_ptr<float>();
    auto btm_diff = btm_out.data_ptr<float>();
    auto weights_diff = weights_out.data_ptr<float>();

    // backward, TODO: add judgement here
    this->weight_gpu_gemm(weights_diff, top_diff, btm_data);
    this->forward_gpu_gemm(btm_diff, top_diff, weights_data);

    return {btm_out, weights_out};
  }

  virtual bool is_deconvolution_layer() { return true; }
};

}  // anonymous namespace

// API implementation
Tensor octree_conv(Tensor data_in, Tensor weights, Tensor octree, int depth,
                   int num_output, vector<int> kernel_size, int stride,
                   bool nempty) {
  OctreeConvOp conv_op(depth, num_output, kernel_size, stride, nempty);
  return conv_op.compute(data_in, weights, octree);
}

Tensor octree_deconv(Tensor data_in, Tensor weights, Tensor octree, int depth,
                     int num_output, vector<int> kernel_size, int stride,
                     bool nempty) {
  OctreeDeconvOp deconv_op(depth, num_output, kernel_size, stride, nempty);
  return deconv_op.compute(data_in, weights, octree);
}

vector<Tensor> octree_conv_grad(Tensor data_in, Tensor weights, Tensor octree,
                                Tensor grad_in, int depth, int num_output,
                                vector<int> kernel_size, int stride,
                                bool nempty) {
  OctreeConvGradOp grad_op(depth, num_output, kernel_size, stride, nempty);
  return grad_op.compute(data_in, weights, octree, grad_in);
}

vector<Tensor> octree_deconv_grad(Tensor data_in, Tensor weights, Tensor octree,
                                  Tensor grad_in, int depth, int num_output,
                                  vector<int> kernel_size, int stride,
                                  bool nempty) {
  OctreeDeconvGradOp grad_op(depth, num_output, kernel_size, stride, nempty);
  return grad_op.compute(data_in, weights, octree, grad_in);
}