#include <octree_nn.h>
#include <octree_parser.h>

#include "ocnn.h"

namespace {

class OctreePadOp {
 public:
  explicit OctreePadOp(int depth, float val = 0.0f)
      : depth_(depth), dval_(val) {
    CHECK_GE(depth_, 1) << "Depth should be larger than 1";
  }

  Tensor compute(Tensor btm_data, Tensor octree) {
    // in octree
    OctreeParser octree_;
    octree_.set_gpu(octree.data_ptr<uint8_t>());

    // btm data
    const float* btm_ptr = btm_data.data_ptr<float>();
    int channel = btm_data.size(1);
    int btm_h = btm_data.size(2);

    // check
    CHECK_EQ(octree_.info().node_num_nempty(depth_), btm_h)
        << ", pad, d = " << depth_ << ", channel = " << channel;

    // top data
    int top_h = octree_.info().node_num(depth_);
    Tensor top_data = torch::zeros({1, channel, top_h, 1}, btm_data.options());
    float* top_ptr = top_data.data_ptr<float>();

    // padding data
    pad_forward_gpu(top_ptr, top_h, channel, btm_ptr, btm_h,
                    octree_.children_gpu(depth_), dval_);
    return top_data;
  }

 protected:
  int depth_;
  float dval_;
};

class OctreeDepadOp {
 public:
  explicit OctreeDepadOp(int depth) : depth_(depth) {
    CHECK_GE(depth_, 1) << "Depth should be larger than 1";
  }

  Tensor compute(Tensor top_data, Tensor octree) {
    // in octree
    OctreeParser octree_;
    octree_.set_gpu(octree.data_ptr<uint8_t>());

    // top grad
    const float* top_ptr = top_data.data_ptr<float>();
    int channel = top_data.size(1);
    int top_h = top_data.size(2);

    // check
    CHECK_EQ(octree_.info().node_num(depth_), top_h)
        << ", depad, d = " << depth_ << ", channel = " << channel;

    // btm grad
    int btm_h = octree_.info().node_num_nempty(depth_);
    Tensor btm_data = torch::zeros({1, channel, btm_h, 1}, top_data.options());
    float* btm_ptr = btm_data.data_ptr<float>();

    // padding data
    pad_backward_gpu(btm_ptr, btm_h, channel, top_ptr, top_h,
                     octree_.children_gpu(depth_));
    return btm_data;
  }

 protected:
  int depth_;
};

}  // anonymous namespace

// API implementation
Tensor octree_pad(Tensor data_in, Tensor octree, int depth, float val) {
  OctreePadOp pad_op(depth, val);
  return pad_op.compute(data_in, octree);
}

Tensor octree_depad(Tensor data_in, Tensor octree, int depth) {
  OctreeDepadOp depad_op(depth);
  return depad_op.compute(data_in, octree);
}
