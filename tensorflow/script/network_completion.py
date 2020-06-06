import tensorflow as tf
from ocnn import *


def get_input_signal(octree, depth, channel):
  signal = octree_property(
      octree, property_name='feature', dtype=tf.float32,
      depth=depth, channel=channel)
  signal = tf.reshape(signal,  [1, channel, -1, 1])
  return signal


def get_split_label(octree, depth):
  label = octree_property(
      octree, property_name='split', dtype=tf.float32,
      depth=depth, channel=1)
  label = tf.reshape(tf.cast(label, dtype=tf.int32), [-1])
  return label


class CompletionResnet:
  def __init__(self, flags):
    self.flags = flags
    self.channels = [4, 512, 512, 256, 128, 64, 32, 16]

  def octree_encoder(self, octree, training, reuse=False):
    flags, channels = self.flags, self.channels
    depth, convd = flags.depth, [None] * 10

    with tf.variable_scope('ocnn_encoder', reuse=reuse):
      with tf.variable_scope('signal_gt'):
        data = get_input_signal(octree, depth, flags.channel)

      with tf.variable_scope("front_%d" % depth):
        convd[depth] = octree_conv_bn_relu(
            data, octree, depth, channels[depth], training)

      for d in range(depth, 1, -1):
        for i in range(0, flags.resblock_num):
          with tf.variable_scope('resblock_%d_%d' % (d, i)):
            convd[d] = octree_resblock(
                convd[d], octree, d, channels[d], 1, training)

        if d > 2:
          with tf.variable_scope('down_%d' % d):
            convd[d-1] = octree_conv_bn_relu(
                convd[d], octree, d, channels[d-1], training,
                stride=2, kernel_size=[2])
    return convd

  def octree_decoder(self, convd, octree0, octree1, training, reuse=False):
    flags, channels = self.flags, self.channels
    depth, deconv = flags.depth, convd[2]
    loss, accu = [], []

    with tf.variable_scope('ocnn_decoder', reuse=reuse):
      for d in range(2, depth + 1):
        if d > 2:
          with tf.variable_scope('up_%d' % d):
            deconv = octree_deconv_bn_relu(
                deconv, octree1, d-1, channels[d], training,
                stride=2, kernel_size=[2])
            if flags.skip_connections:
              skip, _ = octree_align(convd[d], octree0, octree1, d)
              deconv = deconv + skip

          for i in range(0, flags.resblock_num):
            with tf.variable_scope('resblock_%d_%d' % (d, i)):
              deconv = octree_resblock(
                  deconv, octree1, d, channels[d], 1, training)

        with tf.variable_scope('predict_%d' % d):
          logit, label = predict_label(deconv, 2, 32, training)
          # (1, C, H, 1) -> (H, C)
          logit = tf.transpose(tf.squeeze(logit, [0, 3]))

        with tf.variable_scope('loss_%d' % d):
          with tf.variable_scope('label_gt'):
            label_gt = get_split_label(octree1, d)
          loss.append(softmax_loss(logit, label_gt, num_class=2))
          accu.append(label_accuracy(label, label_gt))

        if d == depth:
          with tf.variable_scope('regress_%d' % d):
            signal = predict_signal(deconv, flags.channel, 32, training)

          with tf.variable_scope('loss_regress'):
            with tf.variable_scope('signal_gt'):
              signal_gt = get_input_signal(octree1, depth, flags.channel)
            loss.append(regress_loss(signal, signal_gt))
    return loss, accu

  def decode_shape(self, convd, octree0, training, reuse=False):
    flags, channels = self.flags, self.channels
    depth, deconv = flags.depth, convd[2]

    with tf.variable_scope('ocnn_decoder', reuse=reuse):
      # init the octree
      with tf.variable_scope('octree_0'):
        dis = False if flags.channel < 4 else True
        octree = octree_new(1, channel=flags.channel, has_displace=dis)
      with tf.variable_scope('octree_1'):
        octree = octree_grow(octree, target_depth=1, full_octree=True)
      with tf.variable_scope('octree_2'):
        octree = octree_grow(octree, target_depth=2, full_octree=True)

      for d in range(2, depth + 1):
        if d > 2:
          with tf.variable_scope('up_%d' % d):
            deconv = octree_deconv_bn_relu(
                deconv, octree, d-1, channels[d], training,
                stride=2, kernel_size=[2])
            if flags.skip_connections:
              skip, _ = octree_align(convd[d], octree0, octree, d)
              deconv = deconv + skip

          for i in range(0, flags.resblock_num):
            with tf.variable_scope('resblock_%d_%d' % (d, i)):
              deconv = octree_resblock(
                  deconv, octree, d, channels[d], 1, training)

        with tf.variable_scope('predict_%d' % d):
          _, label = predict_label(deconv, 2, 32, training)

        with tf.variable_scope('octree_%d' % d):
          octree = octree_update(octree, label, depth=d, mask=1)
        if d < depth:
          with tf.variable_scope('octree_%d' % (d+1)):
            octree = octree_grow(octree, target_depth=d+1, full_octree=False)
        else:
          with tf.variable_scope('regress_%d' % d):
            signal = predict_signal(deconv, flags.channel, 32, training)
            signal = normalize_signal(signal)
            signal = octree_mask(signal, label, mask=0)
          with tf.variable_scope('octree_%d' % d):
            octree = octree_set_property(
                octree, signal, property_name="feature", depth=depth)
      return octree
