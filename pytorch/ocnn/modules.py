import torch
import ocnn
import torch.utils.checkpoint

bn_momentum, bn_eps = 0.01, 0.001
# bn_momentum, bn_eps = 0.1, 1e-05


class OctreeConvBn(torch.nn.Module):
  def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1,
               nempty=False):
    super().__init__()
    self.conv = ocnn.OctreeConv(
        depth, channel_in, channel_out, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)

  def forward(self, data_in, octree):
    out = self.conv(data_in, octree)
    out = self.bn(out)
    return out


class OctreeConvBnRelu(torch.nn.Module):
  def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1,
               nempty=False):
    super().__init__()
    self.conv = ocnn.OctreeConv(
        depth, channel_in, channel_out, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data_in, octree):
    out = self.conv(data_in, octree)
    out = self.bn(out)
    out = self.relu(out)
    return out


class OctreeDeConvBnRelu(torch.nn.Module):
  def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1,
               nempty=False):
    super().__init__()
    self.deconv = ocnn.OctreeDeconv(
        depth, channel_in, channel_out, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data_in, octree):
    out = self.deconv(data_in, octree)
    out = self.bn(out)
    out = self.relu(out)
    return out


class FcBnRelu(torch.nn.Module):
  def __init__(self, channel_in, channel_out):
    super().__init__()
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


class OctreeConv1x1(torch.nn.Module):
  def __init__(self, channel_in, channel_out, use_bias=False):
    super().__init__()
    self.conv1x1 = torch.nn.Conv1d(
        channel_in, channel_out, kernel_size=1, bias=use_bias)

  def forward(self, data_in):
    out = torch.squeeze(data_in, dim=-1)  # (1, C, H, 1) -> (1, C, H)
    out = self.conv1x1(out)
    out = torch.unsqueeze(out, dim=-1)  # (1, C, H) -> (1, C, H, 1)
    return out


class OctreeConv1x1Bn(torch.nn.Module):
  def __init__(self, channel_in, channel_out, use_bias=False):
    super().__init__()
    self.conv1x1 = OctreeConv1x1(channel_in, channel_out, use_bias)
    self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)

  def forward(self, data_in):
    out = self.conv1x1(data_in)
    out = self.bn(out)
    return out


class OctreeConv1x1BnRelu(torch.nn.Module):
  def __init__(self, channel_in, channel_out, use_bias=False):
    super().__init__()
    self.conv1x1 = OctreeConv1x1(channel_in, channel_out, use_bias)
    self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data_in):
    out = self.conv1x1(data_in)
    out = self.bn(out)
    out = self.relu(out)
    return out


class OctreeResBlock(torch.nn.Module):
  def __init__(self, depth, channel_in, channel_out, stride=1, bottleneck=4,
               nempty=False):
    super().__init__()
    self.channel_in = channel_in
    self.channel_out = channel_out
    self.bottleneck = bottleneck
    self.stride = stride
    self.depth = depth
    channelb = int(channel_out / bottleneck)
    if self.stride == 2:
      self.maxpool = ocnn.OctreeMaxPool(self.depth)
      self.depth = self.depth - 1
    self.conv1x1a = OctreeConv1x1BnRelu(channel_in, channelb)
    self.conv3x3 = OctreeConvBnRelu(self.depth, channelb, channelb, nempty=nempty)
    self.conv1x1b = OctreeConv1x1Bn(channelb, channel_out)
    if self.channel_in != self.channel_out:
      self.conv1x1c = OctreeConv1x1Bn(channel_in, channel_out)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data_in, octree):
    if self.stride == 2:
      data_in = self.maxpool(data_in, octree)
    conv1 = self.conv1x1a(data_in)
    conv2 = self.conv3x3(conv1, octree)
    conv3 = self.conv1x1b(conv2)
    if self.channel_in != self.channel_out:
      data_in = self.conv1x1c(data_in)
    data_out = self.relu(conv3 + data_in)
    return data_out


class OctreeResBlock2(torch.nn.Module):
  def __init__(self, depth, channel_in, channel_out, stride=1, bottleneck=1,
               nempty=False):
    super().__init__()
    self.channel_in = channel_in
    self.channel_out = channel_out
    self.stride = stride
    self.depth = depth
    channelb = int(channel_out / bottleneck)
    if self.stride == 2:
      self.maxpool = ocnn.OctreeMaxPool(self.depth)
      self.depth = self.depth - 1
    self.conv3x3a = OctreeConvBnRelu(
        self.depth, channel_in, channelb, nempty=nempty)
    self.conv3x3b = OctreeConvBn(
        self.depth, channelb, channel_out, nempty=nempty)
    if self.channel_in != self.channel_out:
      self.conv1x1 = OctreeConv1x1Bn(channel_in, channel_out)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data_in, octree):
    if self.stride == 2:
      data_in = self.maxpool(data_in, octree)
    conv1 = self.conv3x3a(data_in, octree)
    conv2 = self.conv3x3b(conv1, octree)
    if self.channel_in != self.channel_out:
      data_in = self.conv1x1(data_in)
    data_out = self.relu(conv2 + data_in)
    return data_out


class OctreeResBlocks(torch.nn.Module):
  def __init__(self, depth, channel_in, channel_out, resblk_num, bottleneck=4,
               nempty=False, resblk=OctreeResBlock, use_checkpoint=False):
    super().__init__()
    self.resblk_num = resblk_num
    self.use_checkpoint = use_checkpoint
    channels = [channel_in] + [channel_out] * resblk_num
    self.resblks = torch.nn.ModuleList(
        [resblk(depth, channels[i], channels[i+1], 1, bottleneck, nempty)
         for i in range(self.resblk_num)])

  def forward(self, data, octree):
    for i in range(self.resblk_num):
      if self.use_checkpoint:
        data = torch.utils.checkpoint.checkpoint(self.resblks[i], data, octree)
      else:
        data = self.resblks[i](data, octree)
    return data


class OctreeTile(torch.nn.Module):
  ''' This can be regarded as upsampling with the nearest interpolation.
  '''

  def __init__(self, depth):
    super().__init__()
    self.depad = ocnn.OctreeDepad(depth)

  def forward(self, data_in, octree):
    out = self.depad(data_in, octree)
    out = out.repeat(1, 1, 1, 8)
    channel = out.shape[1]
    out = out.view(1, channel, -1, 1)
    return out


def octree_nearest_pts(data, octree, depth, pts, nempty=False):
  key = pts.short()                         # (x, y, z, id)
  key = ocnn.octree_encode_key(key).long()  # (N, )

  idx = ocnn.octree_search_key(key, octree, depth, True, nempty)
  flgs = idx > -1                           # valid indices
  idx = idx * flgs

  data = torch.squeeze(data).t()            # (1, C, H, 1) -> (H, C)
  output = data[idx.long()] * flgs.unsqueeze(-1)
  output = torch.unsqueeze((torch.unsqueeze(output.t(), dim=0)), dim=-1)
  return output


def octree_trilinear_pts(data, octree, depth, pts, nempty=False):
  ''' Linear Interpolatation with input points.
       pts: (N, 4), i.e. N x (x, y, z, id).
      data: (1, C, H, 1)
      nempty: the data only contains features of non-empty octree nodes
  !!! Note: the pts should be scaled into [0, 2^depth]
  '''

  mask = torch.cuda.FloatTensor(
      [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
       [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
  masku = torch.cuda.LongTensor(
      [0, 4294967296, 65536, 4295032832,
       1, 4294967297, 65537, 4295032833])

  # 1. Neighborhood searching
  xyzf, ids = torch.split(pts, [3, 1], 1)
  xyzf = xyzf - 0.5         # the value is defined on the center of each voxel
  xyzi = torch.floor(xyzf)  # the integer part  (N, 3)
  frac = xyzf - xyzi        # the fraction part (N, 3)

  key = torch.cat([xyzi, ids], dim=1).short()  # (N, 4)
  key = ocnn.octree_encode_key(key).long()     # (N, )
  key = (torch.unsqueeze(key, dim=1) + masku).view(-1)  # (N, 1)->(N, 8)->(8*N,)

  idx = ocnn.octree_search_key(key, octree, depth, True, nempty)
  flgs = idx > -1  # valid indices
  idx = idx[flgs]

  # 2. Build the sparse matrix
  npt = pts.shape[0]
  ids = torch.arange(npt).cuda()
  ids = ids.view(-1, 1).repeat(1, 8).view(-1)
  ids = ids[flgs]
  indices = torch.stack([ids, idx], dim=1).long()

  maskc = 1 - mask
  frac = maskc - torch.unsqueeze(frac, dim=1)
  weight = torch.abs(torch.prod(frac, dim=2).view(-1))
  weight = weight[flgs]

  h = data.shape[2]
  mat = torch.sparse.FloatTensor(
      indices.t(), weight, torch.Size([npt, h])).cuda()

  # 3. Interpolatation
  data = torch.squeeze(torch.squeeze(data, dim=0), dim=-1)
  data = torch.transpose(data, 0, 1)
  output = torch.sparse.mm(mat, data)
  norm = torch.sparse.mm(mat, torch.ones(h, 1).cuda())
  output = torch.div(output, norm + 1e-10)
  output = torch.unsqueeze((torch.unsqueeze(output.t(), dim=0)), dim=-1)
  return output


def octree_trilinear(data, octree, depth, target_depth):
  ''' Interpolate data from octree `depth` to `target_depth`
  '''
  xyz = ocnn.octree_property(octree, 'xyz', target_depth)
  xyz = ocnn.octree_decode_key(xyz).float()
  scale = 2.0**(depth-target_depth)
  xyz += torch.cuda.FloatTensor([0.5, 0.5, 0.5, 0.0])
  xyz *= torch.cuda.FloatTensor([scale, scale, scale, 1.0])
  output = octree_trilinear_pts(data, octree, depth, xyz)
  return output


class OctreeTrilinear(torch.nn.Module):
  def __init__(self, depth):
    super().__init__()
    self.depth = depth

  def forward(self, data, octree):
    out = octree_trilinear(data, octree, self.depth, self.depth + 1)
    return out


class OctreeInterp(torch.nn.Module):
  def __init__(self, depth, method='linear', nempty=False):
    super().__init__()
    self.depth = depth
    self.method = method
    self.nempty = nempty

  def forward(self, data, octree, pts):
    # Input pts in [-1, 1], convert pts to [0, 2^depth]
    xyz = (pts[:, :3] + 1.0) * (2 ** (self.depth - 1))
    pts = torch.cat([xyz, pts[:, 3:]], dim=1)

    if self.method == 'nearest':
      out = octree_nearest_pts(data, octree, self.depth, pts, self.nempty)
    elif self.method == 'linear':
      out = octree_trilinear_pts(data, octree, self.depth, pts, self.nempty)
    else:
      raise ValueError
    return out

  def extra_repr(self) -> str:
    return ('depth={}, method={}, nempty={}').format(
            self.depth, self.method, self.nempty)


def create_full_octree(depth, channel, batch_size=1, node_dis=True):
  assert depth > 1
  octree = ocnn.octree_new(batch_size, channel, node_dis)
  for target_depth in range(1, depth+1):
    octree = ocnn.octree_grow(octree, target_depth, full_octree=True)
  return octree


def octree_feature(octree, depth, nempty=False):
  output = ocnn.octree_property(octree, 'feature', depth)
  if nempty:
    output = ocnn.nn.octree_depad(output, octree, depth)
  return output
