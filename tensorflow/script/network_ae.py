import tensorflow as tf
from ocnn import *


class AutoEncoderOcnn:
  def __init__(self, flags):
    self.flags = flags

  def octree_encoder(self, octree, training, reuse=False):
    flags = self.flags
    depth, nout = flags.depth, flags.nout
    channel = [4, nout, 128, 64, 32, 16, 8]
    with tf.variable_scope('ocnn_encoder', reuse=reuse):
      with tf.variable_scope('signal_gt'):
        data = octree_property(octree, property_name="feature", dtype=tf.float32,
                              depth=depth, channel=flags.channel)
        data = tf.reshape(data, [1, flags.channel, -1, 1])
      
      for d in range(depth, 1, -1):
        with tf.variable_scope('depth_%d' % d):
          data = octree_conv_bn_relu(data, octree, d, channel[d], training)
          data, _ = octree_max_pool(data, octree, d)
          
      with tf.variable_scope('depth_1'):
        data = downsample(data, channel[1], training)

      with tf.variable_scope('code'):
        code = conv2d_bn(data, channel[1], kernel_size=1, stride=1, training=training)
        code = tf.nn.tanh(code)
    return code

  def octree_decoder(self, code, octree, training, reuse=False):
    flags = self.flags
    depth = flags.depth
    channel = [512, 256, 128, 64, 32, 16, 8]
    with tf.variable_scope('ocnn_decoder', reuse=reuse):    
      label_gt = [None]*10
      with tf.variable_scope('label_gt'):
        for d in range(2, depth + 1):
          label = octree_property(octree, property_name="split", dtype=tf.float32, 
                                  depth=d, channel=1)
          label_gt[d] = tf.reshape(tf.cast(label, dtype=tf.int32), [-1])

      with tf.variable_scope('signal_gt'):
        signal_gt = octree_property(octree, property_name="feature", dtype=tf.float32, 
                                    depth=depth, channel=flags.channel)
        signal_gt = tf.reshape(signal_gt, [1, flags.channel, -1, 1])

      data = code
      with tf.variable_scope('depth_1'):
        data = upsample(data, channel[1], training)

      loss = []; accu = []; 
      for d in range(2, depth + 1):
        with tf.variable_scope('depth_%d' % d):
          data = octree_upsample(data, octree, d-1, channel[d], training)
          data = octree_conv_bn_relu(data, octree, d, channel[d], training)        
        
        with tf.variable_scope('predict_%d' % d):
          logit, label = predict_label(data, 2, 32, training)

        with tf.variable_scope('loss_%d' % d):
          logit = tf.transpose(tf.squeeze(logit, [0,3])) # (1, C, H, 1) -> (H, C)
          loss.append(softmax_loss(logit, label_gt[d], num_class=2))
          accu.append(label_accuracy(label, label_gt[d]))

        if d == depth:
          with tf.variable_scope('regress_%d' % d):
            signal = predict_signal(data, flags.channel, 32, training)
            loss.append(regress_loss(signal, signal_gt))

    return loss, accu

  def octree_decode_shape(self, code, training, reuse=False):
    flags = self.flags
    depth = flags.depth
    channel = [512, 256, 128, 64, 32, 16, 8]
    with tf.variable_scope('ocnn_decoder', reuse=reuse):
      with tf.variable_scope('octree_0'):
        displace = False if flags.channel < 4 else True
        octree = octree_new(batch_size=1, channel=flags.channel, has_displace=displace)
      with tf.variable_scope('octree_1'):
        octree = octree_grow(octree, target_depth=1, full_octree=True)
      with tf.variable_scope('octree_2'):
        octree = octree_grow(octree, target_depth=2, full_octree=True)

      data = code
      with tf.variable_scope('depth_1'):
        data = upsample(data, channel[1], training)

      for d in range(2, depth + 1):
        with tf.variable_scope('depth_%d' % d):
          data = octree_upsample(data, octree, d-1, channel[d], training)
          data = octree_conv_bn_relu(data, octree, d, channel[d], training)        
        
        with tf.variable_scope('predict_%d' % d):
          _, label = predict_label(data, 2, 32, training)

        with tf.variable_scope('octree_%d' % d, reuse=True):
          octree = octree_update(octree, label, depth=d, mask=1)
          # octree = octree_update(octree, label_gt[d], depth=d, mask=1)
        if d < depth:
          with tf.variable_scope('octree_%d' % (d+1)):
            octree = octree_grow(octree, target_depth=d+1, full_octree=False)
        else:
          with tf.variable_scope('regress_%d' % d):
            signal = predict_signal(data, flags.channel, 32, training)
            signal = normalize_signal(signal)
            signal = octree_mask(signal, label, mask=0)
          with tf.variable_scope('octree_%d' % d, reuse=True):
            octree = octree_set_property(octree, signal, property_name="feature", depth=depth)
    return octree

class AutoEncoderResnet:
  def __init__(self, flags):
    self.flags = flags

  def octree_encoder(self, octree, training, reuse=False):
    flags = self.flags
    depth, nout = flags.depth, flags.nout
    channels = [4, nout, 256, 256, 128, 64, 32, 16]
    with tf.variable_scope('ocnn_encoder', reuse=reuse):
      with tf.variable_scope('signal_gt'):
        data = octree_property(octree, property_name="feature", dtype=tf.float32,
                              depth=depth, channel=flags.channel)
        data = tf.reshape(data, [1, flags.channel, -1, 1])
      
      with tf.variable_scope("front"):
        data = octree_conv_bn_relu(data, octree, depth, channels[depth], training)

      for d in range(depth, 2, -1):
        for i in range(0, flags.resblock_num):
          with tf.variable_scope('resblock_%d_%d' % (d, i)):
            data = octree_resblock(data, octree, d, channels[d], 1, training)
        with tf.variable_scope('down_%d' % d):
          data = octree_conv_bn_relu(data, octree, d, channels[d-1], training,
                                    stride=2, kernel_size=[2])

      with tf.variable_scope('code'):
        # code = conv2d_bn(data, channels[1], kernel_size=1, stride=1, training=training)
        code = octree_conv1x1_bn(data, flags.nout, training=training)
        code = tf.nn.tanh(code)
    return code

  def octree_decoder(self, code, octree, training, reuse=False):
    flags = self.flags    
    depth = flags.depth
    channels = [4, 64, 256, 256, 128, 64, 32, 16]
    with tf.variable_scope('ocnn_decoder', reuse=reuse):    
      data = code
      loss, accu = [], []
      for d in range(2, depth + 1):
        for i in range(0, flags.resblock_num):
          with tf.variable_scope('resblock_%d_%d' % (d, i)):
            data = octree_resblock(data, octree, d, channels[d], 1, training)

        with tf.variable_scope('predict_%d' % d):
          logit, label = predict_label(data, 2, 32, training)
          logit = tf.transpose(tf.squeeze(logit, [0,3])) # (1, C, H, 1) -> (H, C)        

        with tf.variable_scope('loss_%d' % d):
          with tf.variable_scope('label_gt'):
            label_gt = octree_property(octree, property_name="split", 
                dtype=tf.float32, depth=d, channel=1)
            label_gt = tf.reshape(tf.cast(label_gt, dtype=tf.int32), [-1])
          loss.append(softmax_loss(logit, label_gt, num_class=2))
          accu.append(label_accuracy(label, label_gt))

        if d == depth:
          with tf.variable_scope('regress_%d' % d):
            signal = predict_signal(data, flags.channel, 32, training)
          
          with tf.variable_scope('loss_regress'):
            with tf.variable_scope('signal_gt'):
              signal_gt = octree_property(octree, property_name="feature", 
                  dtype=tf.float32, depth=depth, channel=flags.channel)
              signal_gt = tf.reshape(signal_gt, [1, flags.channel, -1, 1])
            loss.append(regress_loss(signal, signal_gt))

        if d < depth:
          with tf.variable_scope('up_%d' % d):
            data = octree_deconv_bn_relu(data, octree, d, channels[d-1], training,
                                        stride=2, kernel_size=[2])
    return loss, accu

  def octree_decode_shape(self, code, training, reuse=False):
    flags = self.flags
    depth = flags.depth
    channels = [4, 64, 256, 256, 128, 64, 32, 16]
    with tf.variable_scope('ocnn_decoder', reuse=reuse):
      with tf.variable_scope('octree_0'):
        displace = False if flags.channel < 4 else True
        octree = octree_new(batch_size=1, channel=flags.channel, has_displace=displace)
      with tf.variable_scope('octree_1'):
        octree = octree_grow(octree, target_depth=1, full_octree=True)
      with tf.variable_scope('octree_2'):
        octree = octree_grow(octree, target_depth=2, full_octree=True)

      data = code
      for d in range(2, depth + 1):
        for i in range(0, flags.resblock_num):
          with tf.variable_scope('resblock_%d_%d' % (d, i)):
            data = octree_resblock(data, octree, d, channels[d], 1, training)
        
        with tf.variable_scope('predict_%d' % d):
          _, label = predict_label(data, 2, 32, training)

        with tf.variable_scope('octree_%d' % d, reuse=True):
          octree = octree_update(octree, label, depth=d, mask=1)
        if d < depth:
          with tf.variable_scope('octree_%d' % (d+1)):
            octree = octree_grow(octree, target_depth=d+1, full_octree=False)
        else:
          with tf.variable_scope('regress_%d' % d):
            signal = predict_signal(data, flags.channel, 32, training)
            signal = normalize_signal(signal)
            signal = octree_mask(signal, label, mask=0)
          with tf.variable_scope('octree_%d' % d, reuse=True):
            octree = octree_set_property(octree, signal, property_name="feature", depth=depth)
        
        if d < depth:
          with tf.variable_scope('up_%d' % d):
            data = octree_deconv_bn_relu(data, octree, d, channels[d-1], training,
                                        stride=2, kernel_size=[2])
    return octree


def make_autoencoder(flags):
  if flags.name == 'ocnn':
    return AutoEncoderOcnn(flags)
  elif flags.name == 'resnet':
    return AutoEncoderResnet(flags)
  else:
    pass
