import os
import sys
import numpy as np
import random
import tensorflow as tf
sys.path.append("..")
from tqdm import tqdm
from config import *
from ocnn import *
from dataset import *

# tf.random.set_random_seed(3)

# octree-based resnet55
def network(octree, depth, num_class, training=True, reuse=None):
  channels = [3, 3, num_class, 256, 128, 64, 32, 16, 8]
  with tf.variable_scope("ocnn_resnet", reuse=reuse):
    data = octree_property(octree, property_name="feature", dtype=tf.float32, 
                           depth=depth, channel=channels[0])
    data = tf.reshape(data, [1, channels[0], -1, 1])

    with tf.variable_scope("conv1"):
      data = octree_conv_bn_relu(data, octree, depth, channels[depth], training)

    for d in range(depth, 2, -1):
      for i in range(1, 7):
        with tf.variable_scope('resblock_%d_%d' % (d, i)):
          data = octree_resblock(data, octree, d, channels[d], 1, training)
      with tf.variable_scope('max_pool_%d' % d):
        data, _ = octree_max_pool(data, octree, d)

    with tf.variable_scope("global_average"):
      data = octree_full_voxel(data, depth=2)
      data = tf.reduce_mean(data, 2)

    with tf.variable_scope("fc2"):
      logit = dense(data, num_class, use_bias=True)

  return logit


# the ocnn in the paper
def network_cls(octree, depth, num_class, training=True, reuse=None):
  channels = [3, num_class, 128, 64, 32, 16, 8, 4]
  with tf.variable_scope("ocnn", reuse=reuse):
    data = octree_property(octree, property_name="feature", dtype=tf.float32,
                          depth=depth, channel=channels[0])
    # data = tf.expand_dims(tf.expand_dims(data, 0), -1)
    data = tf.reshape(data, [1, channels[0], -1, 1])

    for d in range(depth, 2, -1):
      with tf.variable_scope('depth_%d' % d):
        data = octree_conv_bn_relu(data, octree, d, channels[d], training)
        # data, _ = octree_pooling(data, octree, d)
        data, _ = octree_max_pool(data, octree, d)

    with tf.variable_scope("full_voxel"):
      data = octree_full_voxel(data, depth=2)
      data = tf.layers.dropout(data, rate=0.5, training=training)

    with tf.variable_scope("fc1"):
      data = fc_bn_relu(data, channels[2], training=training)
      data = tf.layers.dropout(data, rate=0.5, training=training)

    with tf.variable_scope("fc2"):
      logit = dense(data, num_class, use_bias=True)

  return logit


def loss_functions(logit, label):
  with tf.name_scope('loss'):
    loss = softmax_loss(logit, label, FLAGS.num_class)
    accu = softmax_accuracy(logit, label)
    regularizer = l2_regularizer('ocnn', FLAGS.weight_decay)
  return loss, accu, regularizer


def train_network(reuse=False):
  # octree, label = dataset(FLAGS.train_data, FLAGS.train_batch_size)
  octree, label = points_dataset(FLAGS.train_data, FLAGS.train_batch_size, distort=True)
  logit = network(octree, FLAGS.depth, FLAGS.num_class, training=True, reuse=reuse)
  loss, accu, regularizer = loss_functions(logit, label)
  return loss, accu, regularizer


def test_network(reuse=True):
  octree, label = octree_dataset(FLAGS.test_data, FLAGS.test_batch_size)
  logit = network(octree, FLAGS.depth, FLAGS.num_class, training=False, reuse=reuse)
  loss, accu, regularizer = loss_functions(logit, label)
  return loss, accu, regularizer


def build_solver(total_loss):
  with tf.name_scope('solver'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      lr = learning_rate(global_step)
      solver = tf.train.MomentumOptimizer(lr, 0.9) \
                       .minimize(total_loss, global_step=global_step)
  return solver


def summary_train(loss_train, accu_train, reg_train):
  with tf.name_scope('summary_train'):
    summ_loss_train = tf.summary.scalar("loss_train", loss_train)
    summ_accu_train = tf.summary.scalar("accu_train", accu_train)
    summ_reg_train  = tf.summary.scalar("reg_train", reg_train)
    summ_total_loss = tf.summary.scalar("total_loss", loss_train + reg_train)
    
    summ = tf.summary.merge([summ_loss_train, summ_accu_train,
                             summ_reg_train, summ_total_loss])
  return summ


def summary_test():
  with tf.name_scope('summary_test'):
    loss_test = tf.placeholder(tf.float32)
    accu_test = tf.placeholder(tf.float32)
    reg_test  = tf.placeholder(tf.float32)
    total_loss = loss_test + reg_test
    summ_loss_test = tf.summary.scalar("loss_test", loss_test)
    summ_accu_test = tf.summary.scalar("accu_test", accu_test)
    summ_reg_test  = tf.summary.scalar("reg_test", reg_test)
    summ_total_loss = tf.summary.scalar("total_loss", total_loss)
    
    summ_placeholder = [loss_test, accu_test, reg_test]
    summ = tf.summary.merge([summ_loss_test, summ_accu_test,
                             summ_reg_test, summ_total_loss])
  return summ, summ_placeholder


def train():
  # build graph
  loss_train, accu_train, reg_train = train_network()
  loss_test,  accu_test,  reg_test  = test_network()
  total_loss_train = loss_train + reg_train
  solver = build_solver(total_loss_train)

  # summary
  summ_train = summary_train(loss_train, accu_train, reg_train)
  summ_test, summ_holder = summary_test()

  # checkpoint
  ckpt_path = os.path.join(FLAGS.logdir, 'model')
  ckpt = tf.train.latest_checkpoint(ckpt_path)
  start_iters = 1 if not ckpt else int(ckpt[ckpt.find("iter") + 5:-5])
  tf_saver = tf.train.Saver(max_to_keep=20)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    # initialize or restore
    init = tf.global_variables_initializer()
    sess.run(init)
    if ckpt: tf_saver.restore(sess, ckpt)

    print('Start training ...')
    for i in tqdm(range(start_iters, FLAGS.max_iter + 1)):
      # training
      summary, _ = sess.run([summ_train, solver])
      summary_writer.add_summary(summary, i)

      # testing
      if i % FLAGS.test_every_iter == 0:
        # run testing average
        avg_test = [0, 0, 0]
        for _ in range(FLAGS.test_iter):
          iter_test_result = sess.run([loss_test, accu_test, reg_test])
          for j in range(3):
            avg_test[j] += iter_test_result[j]
        for j in range(3):
          avg_test[j] /= FLAGS.test_iter
        # run testing summary
        summary = sess.run(summ_test, feed_dict=dict(zip(summ_holder, avg_test)))
        summary_writer.add_summary(summary, i)

        # save session
        tf_saver.save(sess, os.path.join(ckpt_path, 'iter_%06d.ckpt' % i))
    print('Training done!')


def test():
  # build graph
  loss_test,  accu_test,  reg_test  = test_network(reuse=False)

  # checkpoint
  assert(FLAGS.ckpt)   # the FLAGS.ckpt should be provided
  tf_saver = tf.train.Saver(max_to_keep=20)

  # start
  avg_test = [0, 0, 0]
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # restore
    tf_saver.restore(sess, FLAGS.ckpt)

    print('Start testing ...')
    for i in range(0, FLAGS.test_iter):
      iter_test_result = sess.run([loss_test, accu_test, reg_test])
      # run testing average
      for j in range(3):
        avg_test[j] += iter_test_result[j]
      print('batch: %04d, loss: %0.4f, reg: %0.4f, accu: %0.4f' % 
            (i, iter_test_result[0], iter_test_result[2], iter_test_result[1]))

  # summary
  print('Testing done!\n')
  for j in range(3):
    avg_test[j] /= FLAGS.test_iter
  print('iter : %04d, loss: %0.4f, reg: %0.4f, accu: %0.4f\n' % 
        (FLAGS.test_iter, avg_test[0], avg_test[2], avg_test[1]))


def main(argv=None):
  if FLAGS.run == "train":
    train()
  else:
    test()


if __name__ == '__main__':
  tf.app.run()
