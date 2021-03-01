import torch
import ocnn


class ResNet(torch.nn.Module):
  def __init__(self, depth, channel_in, nout, resblk_num):
    super(ResNet, self).__init__()
    self.depth, self.channel_in = depth, channel_in
    channels = [2 ** max(11 - i, 2) for i in range(depth + 1)]
    channels.append(channels[depth])

    self.conv1 = ocnn.OctreeConvBnRelu(depth, channel_in, channels[depth])
    self.resblocks = torch.nn.ModuleList(
        [ocnn.OctreeResBlocks(d, channels[d + 1], channels[d], resblk_num)
         for d in range(depth, 2, -1)])
    self.pools = torch.nn.ModuleList(
        [ocnn.OctreeMaxPool(d) for d in range(depth, 2, -1)])
    self.header = torch.nn.Sequential(
        ocnn.FullOctreeGlobalPool(depth=2),      # global pool
        #  torch.nn.Dropout(p=0.5),              # drop
        torch.nn.Linear(channels[3], nout))      # fc

  def forward(self, octree):
    data = ocnn.octree_property(octree, 'feature', self.depth)
    assert data.size(1) == self.channel_in

    data = self.conv1(data, octree)
    for i in range(len(self.resblocks)):
      data = self.resblocks[i](data, octree)
      data = self.pools[i](data, octree)
    data = self.header(data)
    return data
