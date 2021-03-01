import torch
from torch import nn
from torch.autograd import Function

import ocnn


def resize_with_last_val(list_in, num=3):
  assert (type(list_in) is list and len(list_in) < num + 1)
  for i in range(len(list_in), num):
    list_in.append(list_in[-1])
  return list_in


class OctreeDeconvFunction(Function):
  @staticmethod
  def forward(ctx, data_in, weights, octree, depth, channel_out, kernel_size, stride):
    data_in = data_in.contiguous()
    ctx.save_for_backward(data_in, weights, octree)
    ctx.depth = depth
    ctx.channel_out = channel_out
    ctx.kernel_size = resize_with_last_val(kernel_size)
    ctx.stride = stride

    data_out = ocnn.nn.octree_deconv(data_in, weights, octree,
                                     depth, channel_out, kernel_size, stride)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    grad_in = grad_in.contiguous()
    data_in, weights, octree = ctx.saved_tensors
    grad_out, grad_w = ocnn.nn.octree_deconv_grad(data_in, weights, octree,
        grad_in, ctx.depth, ctx.channel_out, ctx.kernel_size, ctx.stride)
    return (grad_out, grad_w) + (None,) * 5


# alias
octree_deconv = OctreeDeconvFunction.apply


# module. TODO: merge code with OctreeConvBase to avoid redundancy
class OctreeDeconvBase(nn.Module):
  def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
    super(OctreeDeconvBase, self).__init__()
    self.depth = depth
    self.channel_out = channel_out
    self.kernel_size = resize_with_last_val(kernel_size)
    self.stride = stride
    self.channel_in = channel_in

    self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    self.dim = channel_out * self.kdim
    self.weights = nn.Parameter(torch.Tensor(self.channel_in, self.dim))
    nn.init.xavier_uniform_(self.weights)

  def extra_repr(self) -> str:
    return 'depth={}, channel_in={}, channel_out={}, kernel_size={}, stride={}'.format(
        self.depth, self.channel_in, self.channel_out, self.kernel_size, self.stride)


class OctreeDeconv(OctreeDeconvBase):
  def forward(self, data, octree):
    if self.stride == 2:
      data = ocnn.octree_depad(data, octree, self.depth)
    deconv = ocnn.octree_deconv(data, self.weights, octree, self.depth,
                                self.channel_out, self.kernel_size, self.stride)
    return deconv

class OctreeDeconvFast(OctreeDeconvBase):
  def forward(self, data, octree):
    depth = self.depth
    if self.stride == 2:
      data = ocnn.octree_depad(data, octree, depth)
      depth = depth + 1
    data = torch.squeeze(torch.squeeze(data, dim=0), dim=-1)
    col = torch.mm(self.weights.t(), data)
    col = col.view(self.channel_out, self.kdim, -1)
    deconv = ocnn.col2octree(col, octree, depth, self.kernel_size, self.stride)
    return deconv