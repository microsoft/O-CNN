import torch
import ocnn

bn_momentum, bn_eps = 0.01, 0.001


class OctreeConvBnRelu(torch.nn.Module):
  def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
    super(OctreeConvBnRelu, self).__init__()
    self.conv = ocnn.OctreeConv(depth, channel_in, channel_out, kernel_size, stride)
    self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data_in, octree):
    out = self.conv(data_in, octree)
    out = self.bn(out)
    out = self.relu(out)
    return out


class FcBnRelu(torch.nn.Module):
  def __init__(self, channel_in, channel_out):
    super(FcBnRelu, self).__init__()
    self.flatten = torch.nn.Flatten(start_dim=1)
    self.fc = torch.nn.Linear(channel_in, channel_out, bias=False)
    self.bn = torch.nn.BatchNorm1d(channel_out, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data_in):
    out = self.flatten(data_in)
    out = self.fc(out)
    out = self.bn(out)
    out = self.relu(out)
    return out
