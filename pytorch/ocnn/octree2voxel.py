import torch
import ocnn


class FullOctree2Voxel(torch.nn.Module):
  def __init__(self, depth):
    super(FullOctree2Voxel, self).__init__()
    self.depth = depth

  def forward(self, data):
    height = 2 ** (3 * self.depth)
    channel = data.size(1)
    out = data.view(channel, -1, height)  # (1, C, H, 1) -> (C, batch_size, H1)
    out = out.permute(1, 0, 2)     # (C, batch_size, H1) -> (batch_size, C, H1)
    return out
  
  def extra_repr(self) -> str:
    return 'depth={}'.format(self.depth)


# TODO: add octree2voxel module for other octree depth
