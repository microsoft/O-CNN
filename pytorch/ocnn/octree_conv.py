import torch
from torch import nn
from torch.autograd import Function

import ocnn


def resize_with_last_val(list_in, num=3):
  assert (type(list_in) is list and len(list_in) < num + 1)
  for i in range(len(list_in), num):
    list_in.append(list_in[-1])
  return list_in


class OctreeConvFunction(Function):
  @staticmethod
  def forward(ctx, data_in, weights, octree, depth, channel_out, kernel_size, stride):
    data_in = data_in.contiguous()
    ctx.save_for_backward(data_in, weights, octree)
    ctx.depth = depth
    ctx.channel_out = channel_out
    ctx.kernel_size = resize_with_last_val(kernel_size)
    ctx.stride = stride

    data_out = ocnn.nn.octree_conv(data_in, weights, octree,
                                   depth, channel_out, kernel_size, stride)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    grad_in = grad_in.contiguous()
    data_in, weights, octree = ctx.saved_tensors
    grad_out, grad_w = ocnn.nn.octree_conv_grad(data_in, weights, octree, grad_in,
        ctx.depth, ctx.channel_out, ctx.kernel_size, ctx.stride)
    return (grad_out, grad_w) + (None,) * 5


# alias
octree_conv = OctreeConvFunction.apply


# module
class OctreeConvBase(nn.Module):
  def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
    super(OctreeConvBase, self).__init__()
    self.depth = depth
    self.channel_out = channel_out
    self.kernel_size = resize_with_last_val(kernel_size)
    self.stride = stride
    self.channel_in = channel_in

    kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    self.dim = channel_in * kdim
    self.weights = nn.Parameter(torch.Tensor(self.channel_out, self.dim))
    nn.init.xavier_uniform_(self.weights)

  def extra_repr(self) -> str:
    return 'depth={}, channel_in={}, channel_out={}, kernel_size={}, stride={}'.format(
        self.depth, self.channel_in, self.channel_out, self.kernel_size, self.stride)


class OctreeConv(OctreeConvBase):
  def forward(self, data_in, octree):
    conv = octree_conv(data_in, self.weights, octree, self.depth,
                       self.channel_out, self.kernel_size, self.stride)
    if self.stride == 2:
      conv = ocnn.octree_pad(conv, octree, self.depth-1)
    return conv


class OctreeConvFast(OctreeConvBase):
  def forward(self, data_in, octree):
    col = ocnn.octree2col(data_in, octree, self.depth,
                          self.kernel_size, self.stride)
    col = col.view([self.dim, -1])
    conv = torch.mm(self.weights, col)
    conv = torch.unsqueeze(torch.unsqueeze(conv, 0), -1)  # [C,H] -> [1,C,H,1]
    if self.stride == 2:
      conv = ocnn.octree_pad(conv, octree, self.depth-1)
    return conv
