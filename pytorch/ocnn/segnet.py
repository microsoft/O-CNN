import torch
import ocnn


class SegNet(torch.nn.Module):
  def __init__(self, depth, channel_in, nout, interp='linear'):
    super(SegNet, self).__init__()
    self.depth, self.channel_in = depth, channel_in
    channels = [2 ** max(10 - i, 2) for i in range(depth + 1)]
    channels.append(channel_in)
    channels[2] = channels[3]
    self.channels = channels

    self.convs = torch.nn.ModuleList(
        [ocnn.OctreeConvBnRelu(d, channels[d + 1], channels[d])
         for d in range(depth, 2, -1)])
    self.pools = torch.nn.ModuleList(
        [ocnn.OctreeMaxPool(d, return_indices=True) for d in range(depth, 2, -1)])

    self.deconvs = torch.nn.ModuleList(
        [ocnn.OctreeConvBnRelu(d, channels[d], channels[d + 1])
         for d in range(2, depth)])
    self.unpools = torch.nn.ModuleList(
        [ocnn.OctreeMaxUnpool(d) for d in range(2, depth)])
    self.deconv = ocnn.OctreeConvBnRelu(depth, channels[depth], channels[depth])

    self.octree_interp = ocnn.OctreeInterp(self.depth, interp, nempty=False)

    self.header = torch.nn.Sequential(
        ocnn.OctreeConv1x1BnRelu(channels[depth], 64),       # fc1
        ocnn.OctreeConv1x1(64, nout, use_bias=True))         # fc2

  def forward(self, octree, pts):
    depth = self.depth
    data = ocnn.octree_feature(octree, depth)
    assert data.size(1) == self.channel_in

    # encoder
    pool_idx = [None] * (depth + 1)
    for i, d in enumerate(range(depth, 2, -1)):
      data = self.convs[i](data, octree)
      data, pool_idx[d] = self.pools[i](data, octree)

    # decoder
    for i, d in enumerate(range(2, depth)):
      data = self.deconvs[i](data, octree)
      data = self.unpools[i](data, pool_idx[d+1], octree)

    # point/voxel feature
    feature = self.deconv(data, octree)
    if pts is not None:
      feature = self.octree_interp(feature, octree, pts)

    # header
    logits = self.header(feature)
    logits = logits.squeeze().t()  # (1, C, H, 1) -> (H, C)
    return logits
