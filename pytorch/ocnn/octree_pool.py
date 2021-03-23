from torch import nn
from torch.autograd import Function

import ocnn


class OctreeMaxPoolFunction(Function):
  @staticmethod
  def forward(ctx, data_in, octree, depth):
    data_in = data_in.contiguous()
    data_out, mask = ocnn.nn.octree_max_pool(data_in, octree, depth)

    ctx.depth = depth
    ctx.save_for_backward(mask, octree)
    return data_out, mask

  @staticmethod
  def backward(ctx, grad_in, mask_grad):
    mask, octree = ctx.saved_tensors
    grad_in = grad_in.contiguous()
    grad_out = ocnn.nn.octree_max_unpool(grad_in, mask, octree, ctx.depth)
    return grad_out, None, None


class OctreeMaxUnpoolFunction(Function):
  @staticmethod
  def forward(ctx, data_in, mask, octree, depth):
    data_in = data_in.contiguous()
    mask = mask.contiguous()
    data_out = ocnn.nn.octree_max_unpool(data_in, mask, octree, depth)

    ctx.depth = depth
    ctx.save_for_backward(mask, octree)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    mask, octree = ctx.saved_tensors
    grad_in = grad_in.contiguous()
    grad_out = ocnn.nn.octree_mask_pool(grad_in, mask, octree, ctx.depth)
    return grad_out, None, None, None


class OctreeMaskPoolFunction(Function):
  @staticmethod
  def forward(ctx, data_in, mask, octree, depth):
    mask = mask.contiguous()
    data_in = data_in.contiguous()
    data_out = ocnn.nn.octree_mask_pool(data_in, mask, octree, depth)

    ctx.depth = depth
    ctx.save_for_backward(mask, octree)
    return data_out

  @staticmethod
  def backward(ctx, grad_in):
    mask, octree = ctx.saved_tensors
    grad_in = grad_in.contiguous()
    grad_out = ocnn.nn.octree_max_unpool(grad_in, mask, octree, ctx.depth)
    return grad_out, None, None, None


# alias
octree_max_pool = OctreeMaxPoolFunction.apply
octree_max_unpool = OctreeMaxUnpoolFunction.apply
octree_mask_pool = OctreeMaskPoolFunction.apply


# module
class OctreePoolBase(nn.Module):
  def __init__(self, depth):
    super(OctreePoolBase, self).__init__()
    self.depth = depth

  def extra_repr(self) -> str:
    return 'depth={}'.format(self.depth)


class OctreeMaxPool(OctreePoolBase):
  def __init__(self, depth, return_indices=False):
    super(OctreeMaxPool, self).__init__(depth)
    self.return_indices = return_indices

  def forward(self, data, octree):
    pool, mask = octree_max_pool(data, octree, self.depth)
    output = ocnn.octree_pad(pool, octree, self.depth-1)  # !!! depth-1
    return output if not self.return_indices else (output, mask)


class OctreeMaxUnpool(OctreePoolBase):
  def forward(self, data, mask, octree):
    pool = ocnn.octree_depad(data, octree, self.depth)  # !!! depth
    output = octree_max_unpool(pool, mask, octree, self.depth + 1)
    return output


class OctreeAvgPool(OctreePoolBase):
  def forward(self, data, octree):
    channel = data.shape[1]
    data = data.view(1, channel, -1, 8)
    pool = data.mean(dim=3, keepdims=True)
    output = ocnn.octree_pad(pool, octree, self.depth-1)  # !!! depth-1
    return output


class FullOctreeGlobalPool(OctreePoolBase):
  def __init__(self, depth):
    super(FullOctreeGlobalPool, self).__init__(depth)
    self.octree2voxel = ocnn.FullOctree2Voxel(depth)

  # for full layer only
  def forward(self, data):
    out = self.octree2voxel(data)
    out = out.mean(dim=2)
    return out
