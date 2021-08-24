from torch import nn
from torch.autograd import Function

import ocnn


class OctreePadFunction(Function):
  @staticmethod
  def forward(ctx, data_in, octree, depth, val=0.0):
    ctx.save_for_backward(octree)
    ctx.depth = depth

    data_in = data_in.contiguous()
    data_out = ocnn.nn.octree_pad(data_in, octree, depth, val)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    octree, = ctx.saved_tensors
    grad_in = grad_in.contiguous()
    grad_out = ocnn.nn.octree_depad(grad_in, octree, ctx.depth)
    return grad_out, None, None, None


class OctreeDepadFunction(Function):
  @staticmethod
  def forward(ctx, data_in, octree, depth):
    ctx.save_for_backward(octree)
    ctx.depth = depth

    data_in = data_in.contiguous()
    data_out = ocnn.nn.octree_depad(data_in, octree, depth)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    octree, = ctx.saved_tensors
    grad_in = grad_in.contiguous()
    grad_out = ocnn.nn.octree_pad(grad_in, octree, ctx.depth, 0.0)
    return grad_out, None, None


# alias
octree_pad = OctreePadFunction.apply
octree_depad = OctreeDepadFunction.apply


# module
class OctreePad(nn.Module):
  def __init__(self, depth, val=0.0):
    super().__init__()
    self.depth = depth
    self.val = val

  def forward(self, data_in, octree):
    return octree_pad(data_in, octree, self.depth, self.val)

  def extra_repr(self) -> str:
    return 'depth={}, val={}'.format(self.depth, self.val)


class OctreeDepad(nn.Module):
  def __init__(self, depth):
    super().__init__()
    self.depth = depth

  def forward(self, data_in, octree):
    return octree_depad(data_in, octree, self.depth)

  def extra_repr(self) -> str:
    return 'depth={}'.format(self.depth)
