import torch
import ocnn
import torch.nn
import torch.nn.functional as F


class OUNet(torch.nn.Module):
  def __init__(self, depth, channel_in, nout, full_depth=2):
    super().__init__()
    self.depth = depth
    self.channel_in = channel_in
    self.nout = nout
    self.full_depth = full_depth
    self.nempty = False
    self.resblk_num = 3
    self.channels = [4, 512, 512, 256, 128, 64, 32, 16]

    # encoder
    self.conv1 = ocnn.OctreeConvBnRelu(
        depth, channel_in, self.channels[depth], nempty=self.nempty)
    self.encoder = torch.nn.ModuleList(
        [ocnn.OctreeResBlocks(d, self.channels[d],
         self.channels[d], self.resblk_num, nempty=self.nempty)
         for d in range(depth, full_depth-1, -1)])
    self.downsample = torch.nn.ModuleList(
        [ocnn.OctreeConvBnRelu(d, self.channels[d],
         self.channels[d-1], kernel_size=[2], stride=2, nempty=self.nempty)
         for d in range(depth, full_depth, -1)])

    # decoder
    self.upsample = torch.nn.ModuleList(
        [ocnn.OctreeDeConvBnRelu(d-1, self.channels[d-1],
         self.channels[d], kernel_size=[2], stride=2, nempty=self.nempty)
         for d in range(full_depth+1, depth + 1)])
    self.decoder = torch.nn.ModuleList(
        [ocnn.OctreeResBlocks(d, self.channels[d],
         self.channels[d], self.resblk_num, nempty=self.nempty)
         for d in range(full_depth+1, depth + 1)])

    # header
    self.predict = torch.nn.ModuleList(
        [self._make_predict_module(self.channels[d], 2)
         for d in range(full_depth, depth + 1)])
    self.header = self._make_predict_module(self.channels[depth], nout)

  def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
    return torch.nn.Sequential(
        ocnn.OctreeConv1x1BnRelu(channel_in, num_hidden),
        ocnn.OctreeConv1x1(num_hidden, channel_out, use_bias=True))

  def get_input_feature(self, octree):
    data = ocnn.octree_property(octree, 'feature', self.depth)
    assert data.size(1) == self.channel_in
    return data

  def ocnn_encoder(self, octree):
    depth, full_depth = self.depth, self.full_depth
    data = self.get_input_feature(octree)

    convs = dict()
    convs[depth] = self.conv1(data, octree)
    for i, d in enumerate(range(depth, full_depth-1, -1)):
      convs[d] = self.encoder[i](convs[d], octree)
      if d > full_depth:
        convs[d-1] = self.downsample[i](convs[d], octree)

    return convs

  def ocnn_decoder(self, convs, octree_out, octree, return_deconvs=False):
    output, deconvs = dict(), dict()
    depth, full_depth = self.depth, self.full_depth

    deconvs[full_depth] = convs[full_depth]
    for i, d in enumerate(range(full_depth, depth+1)):
      if d > full_depth:
        deconvd = self.upsample[i-1](deconvs[d-1], octree_out)
        skip, _ = ocnn.octree_align(convs[d], octree, octree_out, d)
        deconvd = deconvd + skip
        deconvs[d] = self.decoder[i-1](deconvd, octree_out)

      # predict the splitting label
      logit = self.predict[i](deconvs[d])
      logit = logit.squeeze().t()  # (1, C, H, 1) -> (H, C)

      # classification loss
      label_gt = ocnn.octree_property(octree_out, 'split', d).long()
      output['loss_%d' % d] = F.cross_entropy(logit, label_gt)
      output['accu_%d' % d] = logit.argmax(1).eq(label_gt).float().mean()

      if d == depth:
        # predict the signal
        signal = self.header(deconvs[d])
        signal = torch.tanh(signal)

        # regression loss
        signal_gt = ocnn.octree_property(octree_out, 'feature', d)
        output['loss_reg%d' % d] = torch.mean((signal_gt - signal)**2)

    return (output, deconvs) if return_deconvs else output

  def decode_shape(self, convs, octree, return_deconvs=False):
    deconvs = dict()
    depth, full_depth = self.depth, self.full_depth
    octree_out = ocnn.create_full_octree(full_depth, self.nout)

    deconvs[full_depth] = convs[full_depth]
    for i, d in enumerate(range(full_depth, depth+1)):
      if d > full_depth:
        deconvd = self.upsample[i-1](deconvs[d-1], octree_out)
        skip, _ = ocnn.octree_align(convs[d], octree, octree_out, d)
        deconvd = deconvd + skip
        deconvs[d] = self.decoder[i-1](deconvd, octree_out)

      # predict the splitting label
      logit = self.predict[i](deconvs[d])
      logit = logit.squeeze().t()  # (1, C, H, 1) -> (H, C)

      # octree splitting
      label = logit.argmax(1).to(torch.int32)
      octree_out = ocnn.octree_update(octree_out, label, d, split=1)
      if d < depth:
        octree_out = ocnn.octree_grow(octree_out, target_depth=d+1)
      # predict the signal
      else:
        signal = self.header(deconvs[d])  # (1, C, H, 1)
        signal = torch.tanh(signal)
        normal = F.normalize(signal[:, :3], dim=1)
        signal = torch.cat([normal, signal[:, 3:]], dim=1)
        octree_out = ocnn.octree_set_property(octree_out, signal, d)

    return (octree_out, deconvs) if return_deconvs else octree_out

  def forward(self, octree_in, octree_gt=None, run='compute_loss'):
    convs = self.ocnn_encoder(octree_in)
    if 'compute_loss' == run:
      assert octree_gt is not None
      output = self.ocnn_decoder(convs, octree_gt, octree_in)
    elif 'decode_shape' == run:
      output = self.decode_shape(convs, octree_in)
    else:
      raise ValueError
    return output
