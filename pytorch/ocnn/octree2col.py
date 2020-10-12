from torch import nn
from torch.autograd import Function

import ocnn


class Octree2ColFunction(Function):
  @staticmethod
  def forward(ctx, data_in, octree, depth, kernel_size, stride):
    ctx.save_for_backward(octree)
    ctx.depth = depth
    ctx.kernel_size = kernel_size
    ctx.stride = stride

    data_in = data_in.contiguous()
    data_out = ocnn.nn.octree2col(data_in, octree, depth, kernel_size, stride)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    octree, = ctx.saved_tensors
    grad_in = grad_in.contiguous()
    grad_out = ocnn.nn.col2octree(grad_in, octree,
                                  ctx.depth, ctx.kernel_size, ctx.stride)
    return grad_out, None, None, None, None


class Col2OctreeFunction(Function):
  @staticmethod
  def forward(ctx, data_in, octree, depth, kernel_size, stride):
    ctx.save_for_backward(octree)
    ctx.depth = depth
    ctx.kernel_size = kernel_size
    ctx.stride = stride

    data_in = data_in.contiguous()
    data_out = ocnn.nn.col2octree(data_in, octree, depth, kernel_size, stride)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    octree, = ctx.saved_tensors
    grad_in = grad_in.contiguous()
    grad_out = ocnn.nn.octree2col(grad_in, octree,
                                  ctx.depth, ctx.kernel_size, ctx.stride)
    return grad_out, None, None, None, None


# alias
octree2col = Octree2ColFunction.apply
col2octree = Col2OctreeFunction.apply


# module
class Octree2ColBase(nn.Module):
  def __init__(self, depth, kernel_size, stride):
    super(Octree2ColBase, self).__init__()
    self.depth = depth
    self.kernel_size = kernel_size
    self.stride = stride

  def extra_repr(self) -> str:
    return 'depth={}, kernel_size={}, stride={}'.format(
        self.depth, self.kernel_size, self.stride)


class Octree2Col(Octree2ColBase):
  def forward(self, data_in, octree):
    return octree2col(data_in, octree, self.depth, self.kernel_size, self.stride)


class Col2Octree(Octree2ColBase):
  def forward(self, data_in, octree):
    return col2octree(data_in, octree, self.depth, self.kernel_size, self.stride)
