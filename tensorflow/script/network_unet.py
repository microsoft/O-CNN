import tensorflow as tf
from ocnn import *


def network_unet(octree, flags, training, reuse=False):  
  depth = flags.depth
  #nout = [512, 256, 256, 256, 256, 128, 64, 32, 16, 16, 16]
  #nout = [512, 256, 512, 256, 256, 128, 64, 32, 16, 16, 16]
  if depth==7:
    nout = [0,0, 512, 256, 128, 128, 64, 32, 0, 0, 0] 
  else:
    #nout = [  0,   0, 512, 256, 128, 64, 16, 0, 0, 0, 0]
    nout = [  0,   0, 256, 256, 128, 128, 64, 0, 0, 0, 0]

  with tf.variable_scope('ocnn_unet', reuse=reuse):    
    with tf.variable_scope('signal'):
      data = octree_property(octree, property_name='feature', dtype=tf.float32,
                             depth=depth, channel=flags.channel)
      data = tf.reshape(data, [1, flags.channel, -1, 1])

    ## encoder
    convd = [None]*10
    convd[depth+1] = data
    for d in range(depth, 1, -1):
      with tf.variable_scope('encoder_d%d' % d):
        # downsampling
        dd = d if d == depth else d + 1
        stride = 1 if d == depth else 2
        kernel_size = [3] if d == depth else [2]
        convd[d] = octree_conv_bn_relu(convd[d+1], octree, dd, nout[d], training,
                                       stride=stride, kernel_size=kernel_size)
        # resblock
        for n in range(0, flags.resblock_num):
          with tf.variable_scope('resblock_%d' % n):
            convd[d] = octree_resblock(convd[d], octree, d, nout[d], 1, training)

    convd[2] = tf.layers.dropout(convd[2], rate=0.5, training=training)
    ## decoder
    deconv = convd[2]
    for d in range(3, depth + 1):
      with tf.variable_scope('decoder_d%d' % d):
        # upsampling 
        # deconv = octree_tile(deconv, d-1, d, octree=octree1)
        # deconv = octree_upsample(deconv, octree, d-1, nout[d], training)
        deconv = octree_deconv_bn_relu(deconv, octree, d-1, nout[d], training, 
                                       kernel_size=[2], stride=2, fast_mode=False)
        if depth!=7:
          deconv = convd[d] + deconv # skip connections
        elif d % 2==0:
         deconv = convd[d] + deconv # skip connections

        # if d==6:
        #    deconv = tf.layers.dropout(deconv, rate=0.5, training=training)
        # resblock
        for n in range(0, flags.resblock_num):
          with tf.variable_scope('resblock_%d' % n):
            deconv = octree_resblock(deconv, octree, d, nout[d], 1, training)

        # segmentation
        if d == depth:
          with tf.variable_scope('predict_label'):
            logit = predict_module(deconv, flags.nout, 64, training)
            logit = tf.transpose(tf.squeeze(logit, [0, 3])) # (1, C, H, 1) -> (H, C)

  return logit