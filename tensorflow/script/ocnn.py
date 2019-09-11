import sys
import tensorflow as tf
sys.path.append("..")
from libs import *


def get_variables_with_name(name=None, train_only=True, verbose=False):
  if name is None:
    raise Exception("please input a name")

  t_vars = tf.trainable_variables() if train_only else tf.all_variables()
  d_vars = [var for var in t_vars if name in var.name]

  if verbose:
    print("  [*] geting variables with %s" % name)
    for idx, v in enumerate(d_vars):
      print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

  return d_vars



def dense(inputs, nout, use_bias=False):
  inputs = tf.layers.flatten(inputs)
  fc = tf.layers.dense(inputs, nout, use_bias=use_bias, 
      kernel_initializer=tf.contrib.layers.xavier_initializer())
  return fc


def batch_norm(inputs, training, axis=1):
  return tf.layers.batch_normalization(inputs, axis=axis, training=training)


def fc_bn_relu(data, nout, training):
  fc = dense(data, nout)
  bn = batch_norm(fc, training)
  return tf.nn.relu(bn)


def conv2d_bn(data, num_output, kernel_size, stride, training):
  conv = tf.layers.conv2d(data, num_output, kernel_size=kernel_size, 
            strides=stride, data_format="channels_first", use_bias=False, 
            kernel_initializer=tf.contrib.layers.xavier_initializer())
  return batch_norm(conv, training)


def conv2d_bn_relu(data, num_output, kernel_size, stride, training):
  conv = conv2d_bn(data, num_output, kernel_size, stride, training)
  return tf.nn.relu(conv)


def upsample(data, channel, training):
  deconv = tf.layers.conv2d_transpose(data, channel, kernel_size=[8, 1],
      strides=[8, 1], data_format='channels_first', use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer())
  bn = tf.layers.batch_normalization(deconv, axis=1, training=training)
  return tf.nn.relu(bn)


def downsample(data, channel, training):
  deconv = tf.layers.conv2d(data, channel, kernel_size=[8, 1],
      strides=[8, 1], data_format='channels_first', use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer())
  bn = tf.layers.batch_normalization(deconv, axis=1, training=training)
  return tf.nn.relu(bn)


def octree_upsample(data, octree, depth, channel, training):
  if depth > 1:
    data = octree_depad(data, octree, depth)
  return upsample(data, channel, training)


def octree_downsample(data, octree, depth, channel, training):
  down = downsample(data, channel, training)
  return octree_padding(data, octree, depth)


def octree_conv_bn(data, octree, depth, channel, training, fast_mode=False):
  if fast_mode == True:
    conv = octree_conv_fast(data, octree, depth, channel)
  else:
    conv = octree_conv_memory(data, octree, depth, channel)
  return tf.layers.batch_normalization(conv, axis=1, training=training)


def octree_conv_bn_relu(data, octree, depth, channel, training):
  cb = octree_conv_bn(data, octree, depth, channel, training)
  return tf.nn.relu(cb)


def octree_conv_bn_leakyrelu(data, octree, depth, channel, training):
  cb = octree_conv_bn(data, octree, depth, channel, training) 
  return tf.nn.leaky_relu(cb, alpha=0.2)


def octree_resblock(data, octree, depth, num_out, stride, training):
  bottleneck = num_out / 4
  num_in = int(data.shape[1])
  if stride == 2:
    data, mask = octree_max_pool(data, octree, depth=depth)
    depth = depth - 1

  with tf.variable_scope("1x1x1_a"):
    block1 = conv2d_bn(data, bottleneck, stride=1, kernel_size=1, training=training)
    block1 = tf.nn.relu(block1)
  
  with tf.variable_scope("3x3x3"):
    block2 = octree_conv_bn_relu(block1, octree, depth, bottleneck, training)
  
  with tf.variable_scope("1x1x1_b"):
    block3 = conv2d_bn(block2, num_out, stride=1, kernel_size=1, training=training)

  block4 = data
  if num_in != num_out:
    with tf.variable_scope("1x1x1_c"):
      block4 = conv2d_bn(data, num_out, stride=1, kernel_size=1, training=training)

  return tf.nn.relu(block3 + block4)



def predict_module(data, nout, training):
  conv = tf.layers.conv2d(data, 32, kernel_size=1, strides=1,
      data_format='channels_first', use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer())
  conv = tf.layers.batch_normalization(conv, axis=1, training=training)
  conv = tf.nn.relu(conv)
  logit = tf.layers.conv2d(conv, nout, kernel_size=1, strides=1,
      data_format='channels_first', use_bias=True,
      kernel_initializer=tf.contrib.layers.xavier_initializer())
  return logit


def predict_label(data, training):
  logit = predict_module(data, 2, training)
  prob = tf.nn.softmax(logit, axis=1) # logit (1,2,?,1)
  label = tf.argmax(prob, axis=1)     # predict (1,?,1)
  label = tf.reshape(tf.cast(label, tf.int32), [-1])
  return logit, label


def predict_signal(data, channel, training):
  return tf.nn.tanh(predict_module(data, channel, training))


def softmax_loss(logit, label_gt, num_class): 
  label_gt = tf.cast(label_gt, tf.int32)
  onehot = tf.one_hot(label_gt, depth=num_class)
  loss = tf.losses.softmax_cross_entropy(onehot, logit)
  return loss


def l2_regularizer(name, weight_decay):
  with tf.name_scope('l2_regularizer'):
    var = get_variables_with_name(name)
    regularizer = tf.add_n([tf.nn.l2_loss(v) for v in var]) * weight_decay
  return regularizer


def label_accuracy(label, label_gt):
  accuracy = tf.reduce_mean(tf.to_float(tf.equal(label, label_gt)))
  return accuracy


def softmax_accuracy(logit, label): 
  probability = tf.nn.softmax(logit)
  predict = tf.argmax(probability, axis=1)
  return label_accuracy(label, predict)


def regress_loss(signal, signal_gt):
  return tf.reduce_mean(tf.reduce_sum(tf.square(signal-signal_gt), 1))


def normalize(data):
  channel = data.shape[1]
  assert(channel == 3 or channel == 4)
  with tf.variable_scope("normalize"):
    if channel == 4:
      normals = tf.slice(data, [0, 0, 0, 0], [1, 3, -1, 1])
      displacement = tf.slice(data, [0, 3, 0, 0], [1, 1, -1, 1])
      normals = tf.nn.l2_normalize(normals, axis=1)
      output = tf.concat([normals, displacement], axis=1)
    else:
      output = tf.nn.l2_normalize(data, axis=1)
  return output
