import tensorflow as tf
from ocnn import *


def network_segnet(octree, flags, training, reuse=False):
  depth, channel_in = flags.depth, flags.channel
  channels = [2**(9-d) for d in range(0, 8)]
  with tf.variable_scope('ocnn_segnet', reuse=reuse):
    with tf.variable_scope('signal'):
      data = octree_property(octree, property_name='feature', dtype=tf.float32,
                             depth=depth, channel=channel_in)
      data = tf.reshape(data, [1, channel_in, -1, 1])

    # encoder
    unpool_idx = [None]*10
    for d in range(depth, 2, -1):
      with tf.variable_scope('conv_%d' % d):
        data = octree_conv_bn_relu(data, octree, d, channels[d], training)
        data, unpool_idx[d] = octree_max_pool(data, octree, d)

    # # decoder
    with tf.variable_scope('deconv_2'):
      deconv = octree_conv_bn_relu(data, octree, 2, channels[3], training)
    for d in range(3, depth+1):
      with tf.variable_scope('deconv_%d' % d):
        deconv = octree_max_unpool(deconv, unpool_idx[d], octree, d-1)
        deconv = octree_conv_bn_relu(deconv, octree, d, channels[d+1], training)

    # header
    with tf.variable_scope('predict_label'):
      logit = predict_module(deconv, flags.nout, 64, training)
      logit = tf.transpose(tf.squeeze(logit, [0, 3]))  # (1, C, H, 1) -> (H, C)
  return logit
