import os
from ocnn import *
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.client import timeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TFSolver:
  def __init__(self, solver_flags):
    self.flags = solver_flags
    gpus = ','.join([str(a) for a in self.flags.gpu])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

  def build_train_graph(self):
    self.op_train, self.train_tensors, self.test_tensors = None, None, None
    train_names, test_names = [''], ['']
    self.summaries(train_names, train_tensors, test_names)
    
  def summaries(self, train_names, train_tensors, test_names):
    self.summ_train = summary_train(train_names, train_tensors)
    self.summ_test, self.summ_holder = summary_test(test_names)

  def build_test_graph(self):
    self.test_tensors, self.test_names = None, None

  def restore(self, sess, ckpt):
    print('Load checkpoint: ' + ckpt)
    self.tf_saver.restore(sess, ckpt)

  def initialize(self, sess):
    sess.run(tf.global_variables_initializer())

  def train(self):
    # build the computation graph
    self.build_train_graph()

    # checkpoint
    start_iter = 1
    self.tf_saver = tf.train.Saver(max_to_keep=20)
    ckpt_path = os.path.join(self.flags.logdir, 'model')
    if self.flags.ckpt:        # restore from the provided checkpoint
      ckpt = self.flags.ckpt  
    else:                      # restore from the breaking point
      ckpt = tf.train.latest_checkpoint(ckpt_path)    
      if ckpt: start_iter = int(ckpt[ckpt.find("iter")+5:-5]) + 1

    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)

      print('Initialize ...')
      self.initialize(sess)
      if ckpt: self.restore(sess, ckpt)

      print('Start training ...')
      for i in tqdm(range(start_iter, self.flags.max_iter + 1)):
        # training
        summary, _ = sess.run([self.summ_train, self.op_train])
        summary_writer.add_summary(summary, i)

        # testing
        if i % self.flags.test_every_iter == 0:
          # run testing average
          avg_test = run_k_iterations(sess, self.flags.test_iter, self.test_tensors)

          # run testing summary
          summary = sess.run(self.summ_test, 
                             feed_dict=dict(zip(self.summ_holder, avg_test)))
          summary_writer.add_summary(summary, i)

          # save session
          ckpt_name = os.path.join(ckpt_path, 'iter_%06d.ckpt' % i)
          self.tf_saver.save(sess, ckpt_name, write_meta_graph = False)
          
      print('Training done!')

  def timeline(self):
    # build the computation graph
    self.build_train_graph()

    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    timeline_skip, timeline_iter = 10, 2
    with tf.Session(config=config) as sess:
      print('Initialize ...')
      self.initialize(sess)

      print('Start profiling ...')
      for i in tqdm(range(0, timeline_skip + timeline_iter)):
        if i < timeline_skip:
          summary, _ = sess.run([self.summ_train, self.op_train])
        else:
          summary, _ = sess.run([self.summ_train, self.op_train], 
                                options=options, run_metadata=run_metadata)
          if (i == timeline_skip + timeline_iter - 1):
            # summary_writer.add_run_metadata(run_metadata, 'step_%d'%i, i)
            # write timeline to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(os.path.join(self.flags.logdir, 'timeline.json'), 'w') as f:
              f.write(chrome_trace)
        summary_writer.add_summary(summary, i)        
      print('Profiling done!')

  def test(self):
    # build graph
    self.build_test_graph()

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=20)

    # start
    num_tensors = len(self.test_tensors)
    avg_test = [0] * num_tensors
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      # restore and initialize
      self.initialize(sess)
      tf_saver.restore(sess, self.flags.ckpt)

      print('Start testing ...')
      for i in range(0, self.flags.test_iter):
        iter_test_result = sess.run(self.test_tensors)
        # run testing average
        for j in range(num_tensors):
          avg_test[j] += iter_test_result[j]
        # print the results
        reports = 'batch: %04d; ' % i
        for j in range(num_tensors):
          reports += '%s: %0.4f; ' % (self.test_names[j], iter_test_result[j])
        print(reports)

    # summary
    print('Testing done!\n')
    reports = 'batch: %04d; ' % self.flags.test_iter
    for j in range(num_tensors):
      avg_test[j] /= self.flags.test_iter
      reports += '%s: %0.4f; ' % (self.test_names[j], avg_test[j])
    print(reports)
