import sys
import tensorflow as tf
sys.path.append("..")
from libs import *


def get_variables_with_name(name=None, without=None, train_only=True, verbose=False):
  if name is None:
    raise Exception("please input a name")

  t_vars = tf.trainable_variables() if train_only else tf.all_variables()
  d_vars = [var for var in t_vars if name in var.name]

  if without is not None:
    d_vars = [var for var in d_vars if without not in var.name]

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


def fc_bn_relu(inputs, nout, training):
  fc = dense(inputs, nout, use_bias=False)
  bn = batch_norm(fc, training)
  return tf.nn.relu(bn)


def conv2d(inputs, nout, kernel_size, stride, padding='SAME', data_format='channels_first'):
  return tf.layers.conv2d(inputs, nout, kernel_size=kernel_size, strides=stride,
            padding=padding, data_format=data_format, use_bias=False, 
            kernel_initializer=tf.contrib.layers.xavier_initializer())


def conv2d_bn(inputs, nout, kernel_size, stride, training):
  conv = conv2d(inputs, nout, kernel_size, stride)
  return batch_norm(conv, training)


def conv2d_bn_relu(inputs, nout, kernel_size, stride, training):
  conv = conv2d_bn(inputs, nout, kernel_size, stride, training)
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


def avg_pool2d(inputs, data_format='NCHW'):
  return tf.nn.avg_pool2d(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', data_format=data_format)


def global_pool(inputs, data_format='channels_first'):
  axis = [2, 3] if data_format == 'channels_first' else [1, 2]
  return tf.reduce_mean(inputs, axis=axis)
  

def octree_upsample(data, octree, depth, channel, training):
  with tf.variable_scope('octree_upsample'):
    depad = octree_depad(data, octree, depth)
    up = upsample(depad, channel, training)
  return up


def octree_downsample(data, octree, depth, channel, training):
  with tf.variable_scope('octree_downsample'):
    down = downsample(data, channel, training)
    pad = octree_pad(down, octree, depth)
  return pad


def octree_conv_bn(data, octree, depth, channel, training, kernel_size=[3],
                   stride=1, fast_mode=False):
  if fast_mode == True:
    conv = octree_conv_fast(data, octree, depth, channel, kernel_size, stride)
  else:
    conv = octree_conv_memory(data, octree, depth, channel, kernel_size, stride)
  return tf.layers.batch_normalization(conv, axis=1, training=training)


def octree_conv_bn_relu(data, octree, depth, channel, training, kernel_size=[3],
                        stride=1, fast_mode=False):
  with tf.variable_scope('conv_bn_relu'):
    conv_bn = octree_conv_bn(data, octree, depth, channel, training, kernel_size, 
                             stride, fast_mode)
    rl = tf.nn.relu(conv_bn)
  return rl


def octree_conv_bn_leakyrelu(data, octree, depth, channel, training):
  cb = octree_conv_bn(data, octree, depth, channel, training) 
  return tf.nn.leaky_relu(cb, alpha=0.2)


def octree_deconv_bn(data, octree, depth, channel, training, kernel_size=[3],
                     stride=1, fast_mode=False):
  if fast_mode == True:
    conv = octree_deconv_fast(data, octree, depth, channel, kernel_size, stride)
  else:
    conv = octree_deconv_memory(data, octree, depth, channel, kernel_size, stride)
  return tf.layers.batch_normalization(conv, axis=1, training=training)


def octree_deconv_bn_relu(data, octree, depth, channel, training, kernel_size=[3],
                          stride=1, fast_mode=False):
  with tf.variable_scope('deconv_bn_relu'):
    conv_bn = octree_deconv_bn(data, octree, depth, channel, training, kernel_size, 
                               stride, fast_mode)
    rl = tf.nn.relu(conv_bn)
  return rl


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


def octree_resblock2(data, octree, depth, num_out, training):
  # 2 conv layers and stride 1
  with tf.variable_scope("conv_1"):
    conv = octree_conv_bn_relu(data, octree, depth,  num_out/4, training)
  with tf.variable_scope("conv_2"):
    conv = octree_conv_bn(conv, octree, depth, num_out, training)
  out  = tf.nn.relu(conv + data)
  return out


def predict_module(data, num_output, num_hidden, training):
  # MLP with one hidden layer 
  conv = conv2d_bn_relu(data, num_hidden, 1, 1, training)  
  logit = tf.layers.conv2d(conv, num_output, kernel_size=1, strides=1,
      data_format='channels_first', use_bias=True,
      kernel_initializer=tf.contrib.layers.xavier_initializer())
  return logit


def predict_label(data, num_output, num_hidden, training):
  logit = predict_module(data, num_output, num_hidden, training)
  # prob = tf.nn.softmax(logit, axis=1)   # logit   (1, num_output, ?, 1)
  label = tf.argmax(logit, axis=1, output_type=tf.int32)  # predict (1, ?, 1)
  label = tf.reshape(label, [-1]) # flatten
  return logit, label


def predict_signal(data, num_output, num_hidden, training):
  return tf.nn.tanh(predict_module(data, num_output, num_hidden, training))


def softmax_loss(logit, label_gt, num_class): 
  with tf.name_scope('softmax_loss'):
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
  label_gt = tf.cast(label_gt, tf.int32)
  accuracy = tf.reduce_mean(tf.to_float(tf.equal(label, label_gt)))
  return accuracy


def softmax_accuracy(logit, label):
  with tf.name_scope('softmax_accuracy'):
    predict = tf.argmax(logit, axis=1, output_type=tf.int32)
    accu = label_accuracy(predict, tf.cast(label, tf.int32))
  return accu


def regress_loss(signal, signal_gt):
  return tf.reduce_mean(tf.reduce_sum(tf.square(signal-signal_gt), 1))


def normalize_signal(data):
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


def build_solver(total_loss, learning_rate_handle):
  with tf.name_scope('solver'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      lr = learning_rate_handle(global_step)
      solver = tf.train.MomentumOptimizer(lr, 0.9) \
                       .minimize(total_loss, global_step=global_step)
  return solver


def summary_train(names, tensors):
  with tf.name_scope('summary_train'):
    summaries = []
    for it in zip(names, tensors):
      summaries.append(tf.summary.scalar(it[0], it[1]))
    summ = tf.summary.merge(summaries)
  return summ


def summary_test(names):
  with tf.name_scope('summary_test'):
    summaries = []
    summ_placeholder = []
    for name in names:
      summ_placeholder.append(tf.placeholder(tf.float32))
      summaries.append(tf.summary.scalar(name, summ_placeholder[-1]))
    summ = tf.summary.merge(summaries)
  return summ, summ_placeholder


def loss_functions(logit, label_gt, num_class, weight_decay, var_name):
  with tf.name_scope('loss'):
    loss = softmax_loss(logit, label_gt, num_class)
    accu = softmax_accuracy(logit, label_gt)
    regularizer = l2_regularizer(var_name, weight_decay)
  return loss, accu, regularizer


def run_k_iterations(sess, k, tensors):
  num = len(tensors)
  avg_results = [0] * num
  for _ in range(k):
    iter_results = sess.run(tensors)
    for j in range(num):
      avg_results[j] += iter_results[j]
  
  for j in range(num):
    avg_results[j] /= k
  return avg_results