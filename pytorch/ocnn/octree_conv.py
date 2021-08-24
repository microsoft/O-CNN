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
  def forward(ctx, data_in, weights, octree, depth, channel_out, kernel_size,
              stride, nempty):
    data_in = data_in.contiguous()
    ctx.save_for_backward(data_in, weights, octree)
    ctx.depth = depth
    ctx.channel_out = channel_out
    ctx.kernel_size = resize_with_last_val(kernel_size)
    ctx.stride = stride
    ctx.nempty = nempty

    data_out = ocnn.nn.octree_conv(
        data_in, weights, octree, depth, channel_out,
        kernel_size, stride, nempty)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    grad_in = grad_in.contiguous()
    data_in, weights, octree = ctx.saved_tensors
    grad_out, grad_w = ocnn.nn.octree_conv_grad(
        data_in, weights, octree, grad_in, ctx.depth, ctx.channel_out,
        ctx.kernel_size, ctx.stride, ctx.nempty)
    return (grad_out, grad_w) + (None,) * 6


class OctreeDeconvFunction(Function):
  @staticmethod
  def forward(ctx, data_in, weights, octree, depth, channel_out, kernel_size,
              stride, nempty):
    data_in = data_in.contiguous()
    ctx.save_for_backward(data_in, weights, octree)
    ctx.depth = depth
    ctx.channel_out = channel_out
    ctx.kernel_size = resize_with_last_val(kernel_size)
    ctx.stride = stride
    ctx.nempty = nempty

    data_out = ocnn.nn.octree_deconv(
        data_in, weights, octree, depth, channel_out,
        kernel_size, stride, nempty)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    grad_in = grad_in.contiguous()
    data_in, weights, octree = ctx.saved_tensors
    grad_out, grad_w = ocnn.nn.octree_deconv_grad(
        data_in, weights, octree, grad_in, ctx.depth, ctx.channel_out,
        ctx.kernel_size, ctx.stride, ctx.nempty)
    return (grad_out, grad_w) + (None,) * 6


# alias
octree_conv = OctreeConvFunction.apply
octree_deconv = OctreeDeconvFunction.apply


# module
class OctreeConvBase(nn.Module):
  def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1,
               nempty=False):
    super(OctreeConvBase, self).__init__()
    self.depth = depth
    self.channel_out = channel_out
    self.kernel_size = resize_with_last_val(kernel_size)
    self.stride = stride
    self.channel_in = channel_in
    self.nempty = nempty

    self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
    conv_in = channel_in if self.is_conv_layer() else channel_out
    conv_out = channel_out if self.is_conv_layer() else channel_in
    self.cdim = conv_in * self.kdim
    self.weights = nn.Parameter(torch.Tensor(conv_out, self.cdim))
    nn.init.xavier_uniform_(self.weights)

  def is_conv_layer():
    raise NotImplementedError

  def extra_repr(self) -> str:
    return ('depth={}, channel_in={}, channel_out={}, kernel_size={}, '
            'stride={}, nempty={}').format(self.depth, self.channel_in,
             self.channel_out, self.kernel_size, self.stride, self.nempty)


class OctreeConv(OctreeConvBase):
  def is_conv_layer(self):
    return True

  def forward(self, data, octree):
    assert data.size(1) == self.channel_in
    conv = octree_conv(
        data, self.weights, octree, self.depth, self.channel_out,
        self.kernel_size, self.stride, self.nempty)
    if self.stride == 2 and not self.nempty:
      conv = ocnn.octree_pad(conv, octree, self.depth-1)
    return conv


class OctreeDeconv(OctreeConvBase):
  def is_conv_layer(self):
    return False

  def forward(self, data, octree):
    assert data.size(1) == self.channel_in
    if self.stride == 2 and not self.nempty:
      data = ocnn.octree_depad(data, octree, self.depth)
    deconv = octree_deconv(
        data, self.weights, octree, self.depth, self.channel_out,
        self.kernel_size, self.stride, self.nempty)
    return deconv


class OctreeConvFast(OctreeConvBase):
  def is_conv_layer(self):
    return True

  def forward(self, data, octree):
    depth = self.depth
    col = ocnn.octree2col(data, octree, depth, self.kernel_size, self.stride, False)
    col = col.view([self.cdim, -1])
    conv = torch.mm(self.weights, col)
    conv = torch.unsqueeze(torch.unsqueeze(conv, 0), -1)  # [C,H] -> [1,C,H,1]
    if self.stride == 2:
      conv = ocnn.octree_pad(conv, octree, depth-1)
    return conv


class OctreeDeconvFast(OctreeConvBase):
  def is_conv_layer(self):
    return False

  def forward(self, data, octree):
    depth = self.depth
    if self.stride == 2:
      data = ocnn.octree_depad(data, octree, depth)
      depth = depth + 1
    data = torch.squeeze(torch.squeeze(data, dim=0), dim=-1)
    col = torch.mm(self.weights.t(), data)
    col = col.view(self.channel_out, self.kdim, -1)
    deconv = ocnn.col2octree(col, octree, depth, self.kernel_size, self.stride, False)
    return deconv
