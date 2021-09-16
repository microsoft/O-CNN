from torch import nn
from torch.autograd import Function

import ocnn


class OctreeAlignFunction(Function):
  @staticmethod
  def forward(ctx, src_data, src_octree, des_octree, depth):
    src_data = src_data.contiguous()
    des_data, index = ocnn.nn.octree_align(src_data, src_octree, des_octree, depth)
    ctx.save_for_backward(index)
    return des_data, index

  @staticmethod
  def backward(ctx, des_grad, index_grad):
    index, = ctx.saved_tensors
    des_grad = des_grad.contiguous()
    grad_out = ocnn.nn.octree_align_grad(des_grad, index)
    return grad_out, None, None, None


# alias
octree_align = OctreeAlignFunction.apply


# module
class OctreeAlign(nn.Module):
  def __init__(self, depth, return_index=False):
    super().__init__()
    self.depth = depth
    self.return_index = return_index

  def forward(self, src_data, src_octree, des_octree):
    output = octree_align(src_data, src_octree, des_octree, self.depth)
    return output if self.return_index else output[0]

  def extra_repr(self) -> str:
    return 'depth={}'.format(self.depth)
