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


def octree_conv1x1(inputs, nout, use_bias=False):
  outputs = tf.layers.conv2d(inputs, nout, kernel_size=1, strides=1,
                             data_format='channels_first', use_bias=use_bias,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
  return outputs


def octree_conv1x1(inputs, nout, use_bias=False):
  with tf.variable_scope('conv2d_1x1'):
    inputs = tf.squeeze(inputs, axis=[0, 3])   # (1, C, H, 1) -> (C, H)
    weights = tf.get_variable('weights', shape=[nout, int(inputs.shape[0])],
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    outputs = tf.matmul(weights, inputs)       # (C, H) -> (nout, H)
    if use_bias:
      bias = tf.get_variable('bias', shape=[nout, 1], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
      outputs = bias + outputs
    outputs = tf.expand_dims(tf.expand_dims(outputs, axis=0), axis=-1)
  return outputs


def octree_conv1x1_bn(inputs, nout, training):
  conv = octree_conv1x1(inputs, nout, use_bias=False)
  return batch_norm(conv, training)


def octree_conv1x1_bn_relu(inputs, nout, training):
  conv = octree_conv1x1_bn(inputs, nout, training)
  return tf.nn.relu(conv)


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


# !!! Deprecated
def octree_upsample(data, octree, depth, channel, training):
  with tf.variable_scope('octree_upsample'):
    depad = octree_depad(data, octree, depth)
    up = upsample(depad, channel, training)
  return up


def octree_upsample(data, octree, depth, channel, training):
  up = octree_deconv_bn_relu(data, octree, depth, channel, training,
                             kernel_size=[2], stride=2, fast_mode=False)
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
    conv = octree_conv_memory(
        data, octree, depth, channel, kernel_size, stride)
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
    conv = octree_deconv_fast(
        data, octree, depth, channel, kernel_size, stride)
  else:
    conv = octree_deconv_memory(
        data, octree, depth, channel, kernel_size, stride)
  return tf.layers.batch_normalization(conv, axis=1, training=training)


def octree_deconv_bn_relu(data, octree, depth, channel, training, kernel_size=[3],
                          stride=1, fast_mode=False):
  with tf.variable_scope('deconv_bn_relu'):
    conv_bn = octree_deconv_bn(data, octree, depth, channel, training, kernel_size,
                               stride, fast_mode)
    rl = tf.nn.relu(conv_bn)
  return rl


def octree_resblock(data, octree, depth, num_out, stride, training, bottleneck=4):
  num_in = int(data.shape[1])
  channelb = int(num_out / bottleneck)
  if stride == 2:
    data, mask = octree_max_pool(data, octree, depth=depth)
    depth = depth - 1

  with tf.variable_scope("1x1x1_a"):
    block1 = octree_conv1x1_bn_relu(data, channelb, training=training)

  with tf.variable_scope("3x3x3"):
    block2 = octree_conv_bn_relu(block1, octree, depth, channelb, training)

  with tf.variable_scope("1x1x1_b"):
    block3 = octree_conv1x1_bn(block2, num_out, training=training)

  block4 = data
  if num_in != num_out:
    with tf.variable_scope("1x1x1_c"):
      block4 = octree_conv1x1_bn(data, num_out, training=training)

  return tf.nn.relu(block3 + block4)


def octree_resblock2(data, octree, depth, num_out, training):
  num_in = int(data.shape[1])
  with tf.variable_scope("conv_1"):
    conv = octree_conv_bn_relu(data, octree, depth,  num_out/4, training)
  with tf.variable_scope("conv_2"):
    conv = octree_conv_bn(conv, octree, depth, num_out, training)
  
  link = data
  if num_in != num_out:
    with tf.variable_scope("conv_1x1"):
      link = octree_conv1x1_bn(data, num_out, training=training)

  out = tf.nn.relu(conv + link)
  return out


def predict_module(data, num_output, num_hidden, training):
  # MLP with one hidden layer
  with tf.variable_scope('conv1'):
    conv = octree_conv1x1_bn_relu(data, num_hidden, training)
  with tf.variable_scope('conv2'):
    logit = octree_conv1x1(conv, num_output, use_bias=True)
  return logit


def predict_label(data, num_output, num_hidden, training):
  logit = predict_module(data, num_output, num_hidden, training)
  # prob = tf.nn.softmax(logit, axis=1)   # logit   (1, num_output, ?, 1)
  label = tf.argmax(logit, axis=1, output_type=tf.int32)  # predict (1, ?, 1)
  label = tf.reshape(label, [-1])  # flatten
  return logit, label


def predict_signal(data, num_output, num_hidden, training):
  return tf.nn.tanh(predict_module(data, num_output, num_hidden, training))


def softmax_loss(logit, label_gt, num_class, label_smoothing=0.0):
  with tf.name_scope('softmax_loss'):
    label_gt = tf.cast(label_gt, tf.int32)
    onehot = tf.one_hot(label_gt, depth=num_class)
    loss = tf.losses.softmax_cross_entropy(
        onehot, logit, label_smoothing=label_smoothing)
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

    
def average_tensors(tower_tensors):
  avg_tensors = []
  with tf.name_scope('avg_tensors'):
    for tensors in tower_tensors:
      tensors = [tf.expand_dims(tensor, 0) for tensor in tensors]
      avg_tensor = tf.concat(tensors, axis=0)
      avg_tensor = tf.reduce_mean(avg_tensor, 0)
      avg_tensors.append(avg_tensor)
  return avg_tensors


def solver_single_gpu(total_loss, learning_rate_handle, gpu_num=1):
  with tf.variable_scope('solver'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      lr = learning_rate_handle(global_step)
      solver = tf.train.MomentumOptimizer(lr, 0.9) \
                       .minimize(total_loss, global_step=global_step)
  return solver, lr


def solver_multiple_gpus(total_loss, learning_rate_handle, gpu_num):
  tower_grads, variables = [], []
  with tf.device('/cpu:0'):
    with tf.variable_scope('solver'):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      lr = learning_rate_handle(global_step)
      opt = tf.train.MomentumOptimizer(lr, 0.9)

  for i in range(gpu_num):
    with tf.device('/gpu:%d' % i):
      with tf.name_scope('device_b%d' % i):
        grads_and_vars = opt.compute_gradients(total_loss[i])
        grads, variables = zip(*grads_and_vars)
        tower_grads.append(grads)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  # !!! Only get the update_ops defined on `device_0` to avoid the sync 
  # between different GPUs to speed up the training process. !!!
  update_ops = [op for op in update_ops if 'device_0' in op.name]
  assert update_ops, 'The update ops of BN are empty, check the namescope \'device_0\''
  with tf.device('/cpu:0'):
    with tf.name_scope('sync_and_apply_grad'):
      with tf.control_dependencies(update_ops):
        tower_grads = list(zip(*tower_grads))
        avg_grads = average_tensors(tower_grads)
        grads_and_vars = list(zip(avg_grads, variables))
        solver = opt.apply_gradients(grads_and_vars, global_step=global_step)
  return solver, lr


def build_solver(total_loss, learning_rate_handle, gpu_num=1):
  assert (gpu_num > 0)
  the_solver = solver_single_gpu if gpu_num == 1 else solver_multiple_gpus
  return the_solver(total_loss, learning_rate_handle, gpu_num)


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


def loss_functions(logit, label_gt, num_class, weight_decay, var_name, label_smoothing=0.0):
  with tf.name_scope('loss'):
    loss = softmax_loss(logit, label_gt, num_class, label_smoothing)
    accu = softmax_accuracy(logit, label_gt)
    regularizer = l2_regularizer(var_name, weight_decay)
  return [loss, accu, regularizer]


def loss_functions_seg(logit, label_gt, num_class, weight_decay, var_name, mask=-1):
  with tf.name_scope('loss_seg'):
    label_mask = label_gt > mask  # filter label -1
    masked_logit = tf.boolean_mask(logit, label_mask)
    masked_label = tf.boolean_mask(label_gt, label_mask)
    loss = softmax_loss(masked_logit, masked_label, num_class)

    accu = softmax_accuracy(masked_logit, masked_label)
    regularizer = l2_regularizer(var_name, weight_decay)
  return [loss, accu, regularizer]


def get_seg_label(octree, depth):
  with tf.name_scope('seg_label'):
    label = octree_property(octree, property_name='label', dtype=tf.float32,
                            depth=depth, channel=1)
    label = tf.reshape(tf.cast(label, tf.int32), [-1])
  return label


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


def tf_IoU_per_shape(pred, label, class_num, mask=-1):
  with tf.name_scope('IoU'):
    label_mask = label > mask  # filter label -1
    pred = tf.boolean_mask(pred, label_mask)
    label = tf.boolean_mask(label, label_mask)
    pred = tf.argmax(pred, axis=1, output_type=tf.int32)
    IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
    for k in range(class_num):
      pk, lk = tf.equal(pred, k), tf.equal(label, k)
      # pk, lk = pred == k, label == k # why can this not output the right results?
      intsc = tf.reduce_sum(tf.cast(pk & lk, dtype=tf.float32))
      union = tf.reduce_sum(tf.cast(pk | lk, dtype=tf.float32))
      valid = tf.cast(tf.reduce_any(lk), dtype=tf.float32)
      valid_part_num += valid
      IoU += valid * intsc / (union + esp)
    IoU /= valid_part_num + esp
  return IoU, valid_part_num


class Optimizer:
  def __init__(self, stype='SGD', var_list=None, mul=1.0):
    self.stype = stype  # TODO: support more optimizers
    self.mul = mul  # used to modulate the global learning rate
    self.var_list = var_list

  def __call__(self, total_loss, learning_rate):
    with tf.name_scope('solver'):
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = learning_rate(global_step) * self.mul
        solver = tf.train.MomentumOptimizer(lr, 0.9) \
                         .minimize(total_loss, global_step=global_step,
                                   var_list=self.var_list)
    return solver, lr



def octree2points(octree, depth, pts_channel=4, output_normal=False):
  with tf.name_scope('octree2points'):
    signal = octree_signal(octree, depth, 4)    # normal and displacement
    signal = tf.transpose(tf.squeeze(signal, [0, 3]))  # (1, C, H, 1) -> (H, C)
    xyz = octree_xyz(octree, depth)
    xyz = tf.cast(xyz, dtype=tf.float32)

    mask = octree_child(octree, depth) > -1
    signal = tf.boolean_mask(signal, mask)
    xyz = tf.boolean_mask(xyz, mask)

    c = 3.0 ** 0.5 / 2.0
    normal, dis = tf.split(signal, [3, 1], axis=1)
    pts, idx = tf.split(xyz, [3, 1], axis=1)
    pts = (pts + 0.5) + normal * (dis * c)
    if pts_channel == 4:
      pts = tf.concat([pts, idx], axis=1)
    output = pts if not output_normal else (pts, normal)
  return output

