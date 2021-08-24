#include <octree/octree_nn.h>
#include <octree/octree_parser.h>

#include "ocnn.h"

Tensor octree_pad(Tensor data_in, Tensor octree_in, int depth, float val) {
  CHECK_GE(depth, 1) << "Depth should be larger than 1";

  // in octree
  OctreeParser octree_;
  octree_.set_gpu(octree_in.data_ptr<uint8_t>());

  // btm data
  Tensor btm_data = data_in.contiguous();
  const float* btm_ptr = btm_data.data_ptr<float>();
  int channel = btm_data.size(1);
  int btm_h = btm_data.size(2);

  // check
  CHECK_EQ(octree_.info().node_num_nempty(depth), btm_h)
      << ", pad, d = " << depth << ", channel = " << channel;

  // top data
  int top_h = octree_.info().node_num(depth);
  Tensor top_data = torch::zeros({1, channel, top_h, 1}, btm_data.options());
  float* top_ptr = top_data.data_ptr<float>();

  // padding data
  pad_forward_gpu(top_ptr, top_h, channel, btm_ptr, btm_h,
                  octree_.children_gpu(depth), val);
  return top_data;
}

Tensor octree_depad(Tensor data_in, Tensor octree_in, int depth) {
  CHECK_GE(depth, 1) << "Depth should be larger than 1";

  // in octree
  OctreeParser octree_;
  octree_.set_gpu(octree_in.data_ptr<uint8_t>());

  // top grad
  Tensor top_data = data_in.contiguous();
  const float* top_ptr = top_data.data_ptr<float>();
  int channel = top_data.size(1);
  int top_h = top_data.size(2);

  // check
  CHECK_EQ(octree_.info().node_num(depth), top_h)
      << ", depad, d = " << depth << ", channel = " << channel;

  // btm grad
  int btm_h = octree_.info().node_num_nempty(depth);
  Tensor btm_data = torch::zeros({1, channel, btm_h, 1}, top_data.options());
  float* btm_ptr = btm_data.data_ptr<float>();

  // padding data
  pad_backward_gpu(btm_ptr, btm_h, channel, top_ptr, top_h,
                   octree_.children_gpu(depth));
  return btm_data;
}
