import torch
import ocnn


class SegNet(torch.nn.Module):
  def __init__(self, depth, channel_in, nout):
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

    self.header = torch.nn.Sequential(
         ocnn.OctreeConv1x1BnRelu(channels[depth], 64),       # fc1
         ocnn.OctreeConv1x1(64, nout, use_bias=True))         # fc2

  def forward(self, octree):
    depth = self.depth
    data = ocnn.octree_property(octree, 'feature', depth)
    assert data.size(1) == self.channel_in

    pool_idx = [None] * (depth + 1)
    for i, d in enumerate(range(depth, 2, -1)):
      data = self.convs[i](data, octree)
      data, pool_idx[d] = self.pools[i](data, octree)

    for i, d in enumerate(range(2, depth)):
      data = self.deconvs[i](data, octree)
      data = self.unpools[i](data, pool_idx[d+1], octree)
    
    data = self.deconv(data, octree)
    data = self.header(data)
    return data


if __name__ == '__main__':
  from torch.utils.tensorboard import SummaryWriter

  writer = SummaryWriter('logs/segnet')
  octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_2']))
  model = SegNet(depth=5, channel_in=3, nout=4)
  print(model)

  octree = octree.cuda()
  model = model.cuda()
  writer.add_graph(model, octree)
  writer.flush()
