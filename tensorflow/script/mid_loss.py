import tensorflow as tf
from ocnn import softmax_loss, softmax_accuracy

class ShapeLoss:
  def __init__(self, flags, reuse=False):
    self.flags = flags
    self.reuse = reuse

  def _def_memory(self, channel):
    with tf.variable_scope('shape_memory'):
      self.memory = tf.get_variable('memory',
          shape=[self.flags.inst_num, channel], trainable=False,
          initializer=tf.contrib.layers.xavier_initializer())

  def forward(self, feature):
    with tf.variable_scope('shape_cls', reuse=self.reuse):
      self._def_memory(int(feature.shape[1]))
      self.feature = tf.nn.l2_normalize(feature, axis=1)
      logit = tf.matmul(self.feature, self.memory, transpose_a=False, transpose_b=True)
      logit = tf.div(logit, self.flags.sigma)
    return logit

  def loss(self, logit, shape_id):
    self.shape_id = shape_id  # this is the ground-truth label
    with tf.name_scope('shape_loss'):
      loss = softmax_loss(logit, self.shape_id, self.flags.inst_num)
      accu = softmax_accuracy(logit, self.shape_id)
    return loss, accu

  def update_memory(self, solver):
    # update memory bank after solver
    with tf.control_dependencies([solver]):
      with tf.name_scope('update_shape_memory'):
        momentum = self.flags.momentum
        weight = tf.gather(self.memory, self.shape_id)
        weight = self.feature * momentum + weight * (1 - momentum)
        weight = tf.nn.l2_normalize(weight, 1)
        memory = tf.scatter_update(self.memory, self.shape_id, weight)
    return memory

  def knn_accuracy(self, logit, label_test, label_train, class_num=10, K=200):
    with tf.name_scope('knn_accu'):
      one_hot_train = tf.one_hot(label_train, depth=class_num)
      top_k_values, top_k_indices = tf.nn.top_k(logit, k=K)  # k nearest points
      top_k_label = tf.gather(one_hot_train, top_k_indices)  # gather label
      weight = tf.expand_dims(tf.exp(top_k_values), axis=-1) # predict
      weighted_label = tf.multiply(top_k_label, weight)
      sum_up_predictions = tf.reduce_sum(weighted_label, axis=1)
      label_pred = tf.argmax(sum_up_predictions, axis=1)
      accu = label_accuracy(label_pred, label_test)
    return accu


class PointLoss:
  def __init__(self, flags, reuse=False):
    self.flags = flags
    self.reuse = reuse

  def _def_memory(self, channel):
    with tf.variable_scope('point_memory'):
      self.memory = tf.get_variable('memory', trainable=False,
          shape=[self.flags.inst_num, self.flags.seg_num, channel],
          initializer=tf.contrib.layers.xavier_initializer())

  def forward(self, feature, shape_id, obj_segment, batch_size):
    self.shape_id = shape_id
    self.obj_segment = obj_segment
    self.batch_size = batch_size

    with tf.variable_scope('point_cls', reuse=self.reuse):
      self._def_memory(int(feature.shape[1]))
      self.feature = tf.nn.l2_normalize(feature, axis=1)

      # split the feature
      node_nums = tf.segment_sum(tf.ones_like(obj_segment), obj_segment)
      node_nums = tf.reshape(node_nums, [self.batch_size])
      features = tf.split(self.feature, node_nums)

      # gather memory bank
      out = [None] * self.batch_size
      for i in range(self.batch_size):
        out[i] = tf.matmul(features[i], self.memory[shape_id[i], :, :],
                           transpose_a=False, transpose_b=True)

      # logit
      logit = tf.concat(out, axis=0)
      logit = tf.div(logit, self.flags.sigma)
    return logit


  def loss(self, logit, point_id):
    self.point_id = point_id
    with tf.name_scope('point_loss'):
      # point_mask = point_id > -1  # filter label -1
      # logit = tf.boolean_mask(logit, point_mask)
      # point_id = tf.boolean_mask(point_id, point_mask)
      loss = softmax_loss(logit, point_id, self.flags.seg_num)
      accu = softmax_accuracy(logit, point_id)
    return loss, accu


  def update_memory(self, solver):
    # update memory bank after solver
    with tf.control_dependencies([solver]):
      with tf.name_scope('update_point_memory'):
        feature = self.feature
        seg_num, point_id = self.flags.seg_num, self.point_id
        # point_mask = point_id > -1  # filter label -1
        point_id = point_id + (self.obj_segment * seg_num)
        # point_id = tf.boolean_mask(point_id, point_mask)
        # feature = tf.boolean_mask(feature, point_mask)

        batch_size = self.batch_size
        feature = tf.unsorted_segment_mean(feature, point_id, seg_num*batch_size)
        feature = tf.nn.l2_normalize(feature, axis=1)
        feature = tf.reshape(feature, [batch_size, seg_num, -1])

        momentum = self.flags.momentum
        weight = tf.gather(self.memory, self.shape_id)
        weight = feature * momentum + weight * (1 - momentum)
        weight = tf.nn.l2_normalize(weight, axis=2)
        memory = tf.scatter_update(self.memory, self.shape_id, weight)
    return memory
