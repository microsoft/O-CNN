#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

namespace {

Tensor get_ichild(const OctreeParser& octree_, const int depth,
                  const torch::TensorOptions& options) {
  int node_num = octree_.info().node_num(depth);
  int node_num_ne = octree_.info().node_num_nempty(depth);
  const int* child = octree_.children_gpu(depth);
  Tensor tmp = torch::arange(node_num, options.dtype(torch::kInt32));
  Tensor ichild = torch::zeros(node_num_ne, options.dtype(torch::kInt32));
  pad_backward_gpu(ichild.data_ptr<int>(), node_num_ne, 1, tmp.data_ptr<int>(),
                   node_num, child);
  return ichild;
}

class Octree2ColBase {
 public:
  explicit Octree2ColBase(int depth, std::vector<int> kernel_size, int stride)
      : depth_(depth), kernel_size_(kernel_size), stride_(stride) {
    resize_with_last_val(kernel_size_, 3);
    CHECK_GT(depth_, 0) << "Depth should be larger than 0";
    for (auto k : kernel_size_) {
      CHECK(0 < k && k < 4) << "Invalide kernel size";
    }
    CHECK(stride_ == 1 || stride_ == 2) << "Unsupport stride";
  }

  Tensor init_ni_ptr(torch::TensorOptions options) {
    std::vector<int>& ni_cpu = NeighHelper::Get().get_ni(kernel_size_);

    int count = ni_cpu.size();
    Tensor ni_gpu = torch::zeros({count}, options.dtype(torch::kInt32));
    memcpy_gpu(count, ni_cpu.data(), ni_gpu.data_ptr<int>());
    return ni_gpu;
  }

 protected:
  int depth_;
  vector<int> kernel_size_;
  int stride_;
};

class OctreeToColOp : public Octree2ColBase {
 public:
  explicit OctreeToColOp(int depth, std::vector<int> kernel_size, int stride)
      : Octree2ColBase(depth, kernel_size, stride) {}

  Tensor compute(Tensor data_in, Tensor octree_in) {
    // init
    OctreeParser octree_;
    octree_.set_gpu(octree_in.data_ptr<uint8_t>());
    Tensor ni_gpu = this->init_ni_ptr(data_in.options());
    const int* ni_ptr = ni_gpu.data_ptr<int>();

    // input data, data format: [1, channels, H, 1]
    int btm_depth = this->depth_;
    int channel = data_in.size(1);
    int btm_height = data_in.size(2);
    data_in = data_in.contiguous();
    CHECK_EQ(octree_.info().node_num(btm_depth), btm_height);

    // output data
    int top_height = btm_height;
    if (this->stride_ == 2) {
      top_height = btm_height / 8;
      int top_depth = btm_depth - 1;
      CHECK_EQ(top_height, octree_.info().node_num_nempty(top_depth));
    }
    int kernel_sdim = num_elements(this->kernel_size_);
    Tensor data_out =
        torch::zeros({channel, kernel_sdim, top_height}, data_in.options());

    // execute
    octree2col_gpu(data_out.data_ptr<float>(), data_in.data_ptr<float>(),
                   channel, top_height, kernel_sdim, this->stride_,
                   octree_.neighbor_gpu(btm_depth), ni_ptr, top_height, 0);
    return data_out;
  }
};

class ColToOctreeOp : public Octree2ColBase {
 public:
  explicit ColToOctreeOp(int depth, std::vector<int> kernel_size, int stride)
      : Octree2ColBase(depth, kernel_size, stride) {}

  Tensor compute(Tensor grad_in, Tensor octree_in) {
    // init
    OctreeParser octree_;
    octree_.set_gpu(octree_in.data_ptr<uint8_t>());
    Tensor ni_gpu = this->init_ni_ptr(grad_in.options());
    const int* ni_ptr = ni_gpu.data_ptr<int>();

    // in grad shape, data format: [channel, kernel_sdim, top_height]
    int channel = grad_in.size(0);
    int top_height = grad_in.size(2);
    grad_in = grad_in.contiguous();

    // out grad
    int btm_depth = this->depth_;
    int btm_height = octree_.info().node_num(btm_depth);
    if (this->stride_ == 2) {
      CHECK_EQ(top_height, octree_.info().node_num_nempty(btm_depth - 1));
    }
    Tensor grad_out =
        torch::zeros({1, channel, btm_height, 1}, grad_in.options());

    // execute
    int kernel_sdim = num_elements(this->kernel_size_);
    col2octree_gpu(grad_in.data_ptr<float>(), grad_out.data_ptr<float>(),
                   channel, top_height, kernel_sdim, this->stride_,
                   octree_.neighbor_gpu(btm_depth), ni_ptr, top_height, 0);
    return grad_out;
  }
};

class OctreeToColPOp : public Octree2ColBase {
 public:
  explicit OctreeToColPOp(int depth, std::vector<int> kernel_size, int stride)
      : Octree2ColBase(depth, kernel_size, stride) {}

  Tensor compute(Tensor data_in, Tensor octree_in) {
    // init
    OctreeParser octree_;
    torch::TensorOptions options = data_in.options();
    octree_.set_gpu(octree_in.data_ptr<uint8_t>());
    Tensor ni_gpu = this->init_ni_ptr(options);
    const int* ni_ptr = ni_gpu.data_ptr<int>();

    // input data, data format: [1, channels, H, 1]
    int btm_depth = this->depth_;
    int channel = data_in.size(1);
    int btm_height = data_in.size(2);
    data_in = data_in.contiguous();
    int node_num = octree_.info().node_num(btm_depth);
    int node_num_ne = octree_.info().node_num_nempty(btm_depth);
    CHECK_EQ(node_num_ne, btm_height);

    // child pointer
    const int* child = octree_.children_gpu(this->depth_);
    Tensor t0 = torch::arange(node_num, options.dtype(torch::kInt32));
    Tensor t1 = torch::zeros(node_num_ne, options.dtype(torch::kInt32));
    int* ichild = t1.data_ptr<int>();
    pad_backward_gpu(ichild, node_num_ne, 1, t0.data_ptr<int>(), node_num, child);

    // output data
    int top_height = btm_height;
    if (this->stride_ == 2) {
      int top_depth = btm_depth - 1;
      top_height = octree_.info().node_num_nempty(top_depth);
    }
    int kernel_sdim = num_elements(this->kernel_size_);
    Tensor data_out = torch::zeros({channel, kernel_sdim, top_height}, options);

    // execute
    octree2colP_gpu(data_out.data_ptr<float>(), data_in.data_ptr<float>(),
                    channel, top_height, btm_height, kernel_sdim, this->stride_,
                    octree_.neighbor_gpu(btm_depth), ni_ptr, child, ichild,
                    top_height, 0);
    return data_out;
  }
};

class ColToOctreePOp : public Octree2ColBase {
 public:
  explicit ColToOctreePOp(int depth, std::vector<int> kernel_size, int stride)
      : Octree2ColBase(depth, kernel_size, stride) {}

  Tensor compute(Tensor grad_in, Tensor octree_in) {
    // init
    OctreeParser octree_;
    octree_.set_gpu(octree_in.data_ptr<uint8_t>());
    Tensor ni_gpu = this->init_ni_ptr(grad_in.options());
    const int* ni_ptr = ni_gpu.data_ptr<int>();

    // in grad shape, data format: [channel, kernel_sdim, top_height]
    torch::TensorOptions options = grad_in.options();
    int channel = grad_in.size(0);
    int top_height = grad_in.size(2);
    grad_in = grad_in.contiguous();

    // child pointer
    int btm_depth = this->depth_;
    int node_num = octree_.info().node_num(btm_depth);
    int node_num_ne = octree_.info().node_num_nempty(btm_depth);
    const int* child = octree_.children_gpu(this->depth_);
    Tensor t0 = torch::arange(node_num, options.dtype(torch::kInt32));
    Tensor t1 = torch::zeros(node_num_ne, options.dtype(torch::kInt32));
    int* ichild = t1.data_ptr<int>();
    pad_backward_gpu(ichild, node_num_ne, 1, t0.data_ptr<int>(), node_num, child);

    // out grad
    int btm_height = node_num_ne;
    if (this->stride_ == 2) {
      CHECK_EQ(top_height, octree_.info().node_num_nempty(btm_depth - 1));
    } else {
      CHECK_EQ(top_height, node_num_ne);
    }
    Tensor grad_out = torch::zeros({1, channel, btm_height, 1}, grad_in.options());

    // execute
    int kernel_sdim = num_elements(this->kernel_size_);
    col2octreeP_gpu(grad_in.data_ptr<float>(), grad_out.data_ptr<float>(),
                    channel, top_height, btm_height, kernel_sdim, this->stride_,
                    octree_.neighbor_gpu(btm_depth), ni_ptr, child, ichild,
                    top_height, 0);
    return grad_out;
  }
};

}  // anonymous namespace

// API implementation
Tensor octree2col(Tensor data_in, Tensor octree, int depth,
                  std::vector<int> kernel_size, int stride, bool nempty) {
  if (!nempty) {
    OctreeToColOp op(depth, kernel_size, stride);
    return op.compute(data_in, octree);
  } else {
    OctreeToColPOp op(depth, kernel_size, stride);
    return op.compute(data_in, octree);
  }
}

Tensor col2octree(Tensor grad_in, Tensor octree, int depth,
                  std::vector<int> kernel_size, int stride, bool nempty) {
  if (!nempty) {
    ColToOctreeOp op(depth, kernel_size, stride);
    return op.compute(grad_in, octree);
  } else {
    ColToOctreePOp op(depth, kernel_size, stride);
    return op.compute(grad_in, octree);
  }
}
