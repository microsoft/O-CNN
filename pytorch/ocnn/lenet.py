import torch
import ocnn


class LeNet(torch.nn.Module):
  def __init__(self, depth, channel_in, nout):
    super(LeNet, self).__init__()
    self.depth, self.channel_in = depth, channel_in
    channels = [2 ** max(9 - i, 2) for i in range(depth + 1)]
    channels.append(channel_in)

    octree_conv, octree_pool = ocnn.OctreeConvBnRelu, ocnn.OctreeMaxPool
    self.convs = torch.nn.ModuleList(
        [octree_conv(d, channels[d + 1], channels[d]) for d in range(depth, 2, -1)])
    self.pools = torch.nn.ModuleList(
        [octree_pool(d) for d in range(depth, 2, -1)])
    self.octree2voxel = ocnn.FullOctree2Voxel(2)
    self.header = torch.nn.Sequential(
         torch.nn.Dropout(p=0.5),                        # drop1
         ocnn.FcBnRelu(channels[3] * 64, channels[2]),   # fc1
         torch.nn.Dropout(p=0.5),                        # drop2
         torch.nn.Linear(channels[2], nout))             # fc2

  def forward(self, octree):
    data = ocnn.octree_property(octree, 'feature', self.depth)
    assert data.size(1) == self.channel_in
    for i in range(len(self.convs)):
      data = self.convs[i](data, octree)
      data = self.pools[i](data, octree)
    data = self.octree2voxel(data)
    data = self.header(data)
    return data
