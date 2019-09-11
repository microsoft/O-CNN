import os
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
sys.path.append("..")
from config import *
from ocnn import *


def octree_encoder(octree, depth, nout, training, reuse=False):
  channel = [4, nout, 128, 64, 32, 16, 8]
  with tf.variable_scope('octree_encoder', reuse=reuse):
    with tf.variable_scope('signal_gt'):
      data = octree_property(octree, property_name="feature", dtype=tf.float32,
                             depth=depth, channel=FLAGS.channel)
      data = tf.reshape(data, [1, FLAGS.channel, -1, 1])
    
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


def octree_decoder(code, octree, depth, training, reuse=False):
  channel = [512, 256, 128, 64, 32, 16, 8]
  with tf.variable_scope('octree_decoder', reuse=reuse):    
    label_gt = [None]*10
    with tf.variable_scope('label_gt'):
      for d in range(2, depth + 1):
        label = octree_property(octree, property_name="split", dtype=tf.float32, 
                                depth=d, channel=1)
        label_gt[d] = tf.reshape(tf.cast(label, dtype=tf.int32), [-1])

    with tf.variable_scope('signal_gt'):
      signal_gt = octree_property(octree, property_name="feature", dtype=tf.float32, 
                                  depth=depth, channel=FLAGS.channel)
      signal_gt = tf.reshape(signal_gt, [1, FLAGS.channel, -1, 1])

    data = code
    with tf.variable_scope('depth_1'):
      data = upsample(data, channel[1], training)

    loss = []; accu = []; 
    for d in range(2, depth + 1):
      with tf.variable_scope('depth_%d' % d):
        data = octree_upsample(data, octree, d-1, channel[d], training)
        data = octree_conv_bn_relu(data, octree, d, channel[d], training)        
      
      with tf.variable_scope('predict_%d' % d):
        logit, label = predict_label(data, training)

      with tf.variable_scope('loss_%d' % d):
        logit = tf.transpose(tf.squeeze(logit, [0,3])) # (1, C, H, 1) -> (H, C)
        loss.append(softmax_loss(logit, label_gt[d], num_class=2))
        accu.append(label_accuracy(label, label_gt[d]))

      if d == depth:
        with tf.variable_scope('regress_%d' % d):
          loss.append(regress_loss(predict_signal(data, FLAGS.channel, training), signal_gt))

  return loss, accu


def octree_decode_shape(code, depth, training, reuse=False):
  channel = [512, 256, 128, 64, 32, 16, 8]

  with tf.variable_scope('octree_decoder', reuse=reuse):
    with tf.variable_scope('octree_0'):
      displace = False if FLAGS.channel < 4 else True
      octree = octree_new(batch_size=1, channel=FLAGS.channel, has_displace=displace)
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
        _, label = predict_label(data, training)

      with tf.variable_scope('octree_%d' % d, reuse=True):
        octree = octree_update(octree, label, depth=d, mask=1)
        # octree = octree_update(octree, label_gt[d], depth=d, mask=1)
      if d < depth:
        with tf.variable_scope('octree_%d' % (d+1)):
          octree = octree_grow(octree, target_depth=d+1, full_octree=False)
      else:
        with tf.variable_scope('regress_%d' % d):
          signal = predict_signal(data, FLAGS.channel, training)
          signal = normalize(signal)
          signal = octree_mask(signal, label, mask=0)
        with tf.variable_scope('octree_%d' % d, reuse=True):
          octree = octree_set_property(octree, signal, property_name="feature", depth=depth)

  return octree


def train_network():
  octree, label = dataset(FLAGS.train_data, FLAGS.train_batch_size)

  code = octree_encoder(octree, FLAGS.depth, nout=128, training=True, reuse=False)
  loss, accu = octree_decoder(code, octree, FLAGS.depth, training=True, reuse=False)

  with tf.name_scope('compute_loss'):
    var_all = tf.trainable_variables()
    regularizer = tf.add_n([tf.nn.l2_loss(v) for v in var_all]) * FLAGS.weight_decay
    total_loss  = tf.add_n(loss + [regularizer])

  with tf.name_scope('solver'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      lr = learning_rate(global_step)
      solver = tf.train.MomentumOptimizer(lr, 0.9) \
                       .minimize(total_loss, global_step=global_step)

  with tf.name_scope('summary_train'):
    summaries = []
    summary_names = ['loss%d' % d for d in range(2, FLAGS.depth + 1)] + ['normal'] + \
                    ['accu%d' % d for d in range(2, FLAGS.depth + 1)] + ['total_loss']
    for item in zip(summary_names, loss + accu + [total_loss]):
      summaries.append(tf.summary.scalar(item[0], item[1]))
    train_summary = tf.summary.merge(summaries)

  return train_summary, solver 


def test_network():
  octree, label = dataset(FLAGS.train_data, FLAGS.train_batch_size)
  
  code = octree_encoder(octree, FLAGS.depth, nout=128, training=False, reuse=True)
  loss, accu = octree_decoder(code, octree, FLAGS.depth, training=False, reuse=True)
  test_output = loss + accu

  with tf.name_scope('summary_test'):
    summaries = []; average_test = [];
    summary_names = ['loss%d' % d for d in range(2, FLAGS.depth + 1)] + ['regress'] + \
                    ['accu%d' % d for d in range(2, FLAGS.depth + 1)]
    for name in summary_names:
      hd = tf.placeholder(tf.float32)
      average_test.append(hd)
      summaries.append(tf.summary.scalar(name, hd))
    test_summary = tf.summary.merge(summaries)

  return test_summary, average_test, test_output


def train():
  train_summary, solver = train_network()
  test_summary, average_test, test_output = test_network()
  tf_saver = tf.train.Saver(max_to_keep=200)
  
  start_iter = 1
  model_dir = os.path.join(FLAGS.logdir, 'model')
  ckpt = tf.train.latest_checkpoint(model_dir)
  if ckpt is not None: 
    start_iter = int(ckpt[ckpt.find('iter') + 4:-5])
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    print('begin training...')
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    print('\nstart_iter = %d\n' % start_iter)
    if start_iter != 1:
      tf_saver.restore(sess, ckpt)
      print('\nload ckpt %d done!\n' % start_iter)

    for i in tqdm(range(start_iter, FLAGS.max_iter+1)):
      # if i % 10 == 0: print('training at step - ', i)
      summary, _ = sess.run([train_summary, solver])
      summary_writer.add_summary(summary, i)

      if i % FLAGS.test_every_iter == 0:
        # print('testing at step - %d'%i)  
        output_num = len(test_output)
        avg_value_test = [0] * output_num        
        for test_i in range(FLAGS.test_iter):
          iter_test_output = sess.run(test_output)
          for j in range(output_num):                      
            avg_value_test[j] += iter_test_output[j]
        for j in range(output_num):
          avg_value_test[j] /= FLAGS.test_iter
        
        feed_dict = dict(zip(average_test, avg_value_test))
        summary = sess.run(test_summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary, i) 
        tf_saver.save(sess, os.path.join(model_dir, 'iter_%06d.ckpt' % i))


def test():
  octree_gt, label = dataset(FLAGS.test_data, 1)

  code = octree_encoder(octree_gt, FLAGS.depth, nout=128, training=False, reuse=False)
  octree = octree_decode_shape(code, FLAGS.depth, training=False, reuse=False)

  ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
  assert(ckpt is not None)
  print('testing with ', ckpt)
  tf_saver = tf.train.Saver()
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    tf_saver.restore(sess, ckpt)
    tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    for i in range(FLAGS.test_iter):
      print("iter: ", i)
      origin, reconstructed = sess.run([octree_gt, octree])
      with open(FLAGS.logdir + ('/%04d_input.octree' % i), "wb") as f:
        f.write(origin.tobytes())
      with open(FLAGS.logdir + ('/%04d_output.octree' % i), "wb") as f:
        f.write(reconstructed.tobytes())



def main(argv=None):
  if FLAGS.run == "train":
    train()
  else:
    test()

if __name__ == '__main__':
  tf.app.run()
