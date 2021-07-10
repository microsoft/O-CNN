import tensorflow as tf
from ocnn import *


def network_aenet(octree, flags, training, reuse=False):  

  depth, nout = flags.depth, flags.nout
  channel = [None, None, None, 256, 128, 32, 16]
  cout=[None, None, 256, 128, 32, 16, 4]

  with tf.compat.v1.variable_scope('ocnn_unet', reuse=reuse):    
    with tf.compat.v1.variable_scope('signal'):
      data = octree_property(octree, property_name='feature', dtype=tf.float32,
                             depth=depth, channel=flags.channel)
      data = tf.reshape(data, [1, flags.channel, -1, 1])

    ## encoder
    mask = [None]*10
    for d in range(depth, 2, -1):
    #for d in range(depth, 5, -1):
      with tf.compat.v1.variable_scope('encoder_d%d' % d):
          # data=tf.Print(data,[tf.shape(data)],"step a",summarize=10) 
          data = octree_conv_bn_relu(data, octree, d, channel[d], training)
          if d==5:
             data = tf.compat.v1.layers.dropout(data, rate=0.5, training=training)
          # data=tf.Print(data,[tf.shape(data)],"step b",summarize=10) 
          data,mask[d] = octree_max_pool(data, octree, d)
          # data=tf.Print(data,[tf.shape(data)],"step c",summarize=10) 


    ## decoder
    assert d != 2 ,print("trouble 1")
    data = octree_conv_bn_relu(data, octree, 2, 256, training)
    data = tf.compat.v1.layers.dropout(data, rate=0.5, training=training)
    
    for d in range(3, depth + 1):
    #for d in range(6, depth + 1):
      with tf.compat.v1.variable_scope('decoder_d%d' % (d-1)):
        # mask[d]=tf.Print(mask[d],[tf.shape(mask[d])],"step mask",summarize=10)
        # data=tf.Print(data,[tf.shape(data)],"step 1",summarize=10)
        data=  octree_max_unpool(data, mask[d], octree, d-1)
        #data=tf.Print(data,[tf.shape(data)],"step 2",summarize=10)
        data = octree_conv_bn_relu(data, octree, d, cout[d], training) 
        if d==5:
            data = tf.compat.v1.layers.dropout(data, rate=0.5, training=training)
        #data=tf.Print(data,[tf.shape(data)],"step 3",summarize=10)  

        # segmentation
        if d == depth:
          with tf.compat.v1.variable_scope('predict_label'):
            logit = predict_module(data, flags.nout, 64, training)
            logit = tf.transpose(a=tf.squeeze(logit, [0, 3])) # (1, C, H, 1) -> (H, C)

  return logit

          # downsampling
        # dd = d if d == depth else d + 1
        # stride = 1 if d == depth else 2
        # kernel_size = [3] if d == depth else [2]
        # convd[d] = octree_conv_bn_relu(convd[d+1], octree, dd, nout[d], training,
        #                                stride=stride, kernel_size=kernel_size)
        # # resblock
        # for n in range(0, flags.resblock_num):
        #   with tf.variable_scope('resblock_%d' % n):
        #     convd[d] = octree_resblock(convd[d], octree, d, nout[d], 1, training)
        # upsampling
        # deconv = octree_tile(deconv, d-1, d, octree=octree1)
        # deconv = octree_upsample(deconv, octree, d-1, nout[d], training)
        # deconv = octree_deconv_bn_relu(deconv, octree, d-1, nout[d], training, 
        #                                kernel_size=[2], stride=2, fast_mode=False)
        # deconv = convd[d] + deconv # skip connections

        # # resblock
        # for n in range(0, flags.resblock_num):
        #   with tf.variable_scope('resblock_%d' % n):
        #     deconv = octree_resblock(deconv, octree, d, nout[d], 1, training)