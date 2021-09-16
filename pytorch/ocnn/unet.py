import torch
import ocnn
import torch.nn


class UNet(torch.nn.Module):
  def __init__(self, depth, channel_in, nout, nempty=False, interp='linear',
               use_checkpoint=False):
    super(UNet, self).__init__()
    self.depth = depth
    self.channel_in = channel_in
    self.nempty = nempty
    self.use_checkpoint = use_checkpoint
    self.config_network()
    self.stages = len(self.encoder_blocks)

    # encoder
    self.conv1 = ocnn.OctreeConvBnRelu(
        depth, channel_in, self.encoder_channel[0], nempty=nempty)
    self.downsample = torch.nn.ModuleList(
        [ocnn.OctreeConvBnRelu(depth - i, self.encoder_channel[i],
         self.encoder_channel[i+1], kernel_size=[2], stride=2, nempty=nempty)
         for i in range(self.stages)])
    self.encoder = torch.nn.ModuleList(
        [ocnn.OctreeResBlocks(depth - i - 1, self.encoder_channel[i+1],
         self.encoder_channel[i+1], self.encoder_blocks[i], self.bottleneck,
         nempty, self.resblk, self.use_checkpoint) for i in range(self.stages)])

    # decoder
    depth = depth - self.stages
    channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
               for i in range(self.stages)]
    self.upsample = torch.nn.ModuleList(
        [ocnn.OctreeDeConvBnRelu(depth + i, self.decoder_channel[i],
         self.decoder_channel[i+1], kernel_size=[2], stride=2, nempty=nempty)
         for i in range(self.stages)])
    self.decoder = torch.nn.ModuleList(
        [ocnn.OctreeResBlocks(depth + i + 1, channel[i],
         self.decoder_channel[i+1], self.decoder_blocks[i], self.bottleneck,
         nempty, self.resblk, self.use_checkpoint) for i in range(self.stages)])

    # interpolation
    self.octree_interp = ocnn.OctreeInterp(self.depth, interp, nempty)

    # header
    self.header = self.make_predict_module(self.decoder_channel[-1], nout)

  def config_network(self):
    self.encoder_channel = [32, 32, 64, 128, 256]
    self.decoder_channel = [256, 256, 128, 96, 96]
    self.encoder_blocks = [2, 3, 4, 6]
    self.decoder_blocks = [2, 2, 2, 2]
    self.bottleneck = 1
    self.resblk = ocnn.OctreeResBlock2

  def make_predict_module(self, channel_in, channel_out=2, num_hidden=64):
    return torch.nn.Sequential(
        ocnn.OctreeConv1x1BnRelu(channel_in, num_hidden),
        ocnn.OctreeConv1x1(num_hidden, channel_out, use_bias=True))

  def forward(self, octree, pts=None):
    depth = self.depth
    data = ocnn.octree_feature(octree, depth, self.nempty)
    assert data.size(1) == self.channel_in

    # encoder
    convd = [None] * 16
    convd[depth] = self.conv1(data, octree)
    stages = len(self.encoder_blocks)
    for i in range(stages):
      depth_i = depth - i - 1
      conv = self.downsample[i](convd[depth_i+1], octree)
      convd[depth_i] = self.encoder[i](conv, octree)

    # decoder
    depth = depth - stages
    deconv = convd[depth]
    for i in range(stages):
      depth_i = depth + i + 1
      deconv = self.upsample[i](deconv, octree)
      deconv = torch.cat([convd[depth_i], deconv], dim=1)  # skip connections
      deconv = self.decoder[i](deconv, octree)

    # point/voxel feature
    feature = deconv
    if pts is not None:
      feature = self.octree_interp(feature, octree, pts)

    # header
    logits = self.header(feature)
    logits = logits.squeeze().t()  # (1, C, H, 1) -> (H, C)
    return logits
