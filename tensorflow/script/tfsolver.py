import os
from ocnn import *
from tqdm import tqdm
import tensorflow as tf
from learning_rate import LRFactory
from tensorflow.python.client import timeline

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TFSolver:
  def __init__(self, flags, compute_graph=None, build_solver=build_solver):
    self.flags = flags.SOLVER
    self.graph = compute_graph
    self.build_solver = build_solver

  def build_train_graph(self):
    gpu_num = len(self.flags.gpu)
    train_params = {'dataset': 'train', 'training': True,  'reuse': False}
    test_params  = {'dataset': 'test',  'training': False, 'reuse': True}
    if gpu_num > 1:
      train_params['gpu_num'] = gpu_num
      test_params['gpu_num']  = gpu_num
      
    self.train_tensors, train_names = self.graph(**train_params)
    self.test_tensors, self.test_names = self.graph(**test_params)
    
    total_loss = self.train_tensors[train_names.index('total_loss')]
    solver_param = [total_loss, LRFactory(self.flags)]
    if gpu_num > 1:
      solver_param.append(gpu_num)
    self.train_op, lr = self.build_solver(*solver_param)

    if gpu_num > 1: # average the tensors from different gpus for summaries
      with tf.device('/cpu:0'):
        self.train_tensors = average_tensors(self.train_tensors)
        self.test_tensors = average_tensors(self.test_tensors)        
    self.summaries(train_names + ['lr'], self.train_tensors + [lr,], self.test_names)

  def summaries(self, train_names, train_tensors, test_names):
    self.summ_train = summary_train(train_names, train_tensors)
    self.summ_test, self.summ_holder = summary_test(test_names)
    self.summ2txt(test_names, 'step', 'w')

  def summ2txt(self, values, step, flag='a'):
    test_summ = os.path.join(self.flags.logdir, 'test_summaries.csv')
    with open(test_summ, flag) as fid:      
      msg = '{}'.format(step)
      for v in values:
        msg += ', {}'.format(v)
      fid.write(msg + '\n')

  def build_test_graph(self):
    gpu_num = len(self.flags.gpu)
    test_params  = {'dataset': 'test',  'training': False, 'reuse': False}
    if gpu_num > 1: test_params['gpu_num'] = gpu_num
    self.test_tensors, self.test_names = self.graph(**test_params)
    if gpu_num > 1: # average the tensors from different gpus
      with tf.device('/cpu:0'):
        self.test_tensors = average_tensors(self.test_tensors)        

  def restore(self, sess, ckpt):
    print('Load checkpoint: ' + ckpt)
    self.tf_saver.restore(sess, ckpt)

  def initialize(self, sess):
    sess.run(tf.global_variables_initializer())

  def run_k_iterations(self, sess, k, tensors):
    num = len(tensors)
    avg_results = [0] * num
    for _ in range(k):
      iter_results = sess.run(tensors)
      for j in range(num):
        avg_results[j] += iter_results[j]
    
    for j in range(num):
      avg_results[j] /= k
    avg_results = self.result_callback(avg_results)
    return avg_results

  def result_callback(self, avg_results):
    return avg_results # calc some metrics, such as IoU, based on the graph output

  def train(self):
    # build the computation graph
    self.build_train_graph()

    # checkpoint
    start_iter = 1
    self.tf_saver = tf.train.Saver(max_to_keep=self.flags.ckpt_num)
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
      for i in tqdm(range(start_iter, self.flags.max_iter + 1), ncols=80):
        # training
        summary, _ = sess.run([self.summ_train, self.train_op])
        summary_writer.add_summary(summary, i)

        # testing
        if i % self.flags.test_every_iter == 0:
          # run testing average
          avg_test = self.run_k_iterations(sess, self.flags.test_iter, self.test_tensors)

          # run testing summary
          summary = sess.run(self.summ_test, 
                             feed_dict=dict(zip(self.summ_holder, avg_test)))
          summary_writer.add_summary(summary, i)
          self.summ2txt(avg_test, i)

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
    timeline_skip, timeline_iter = 100, 2
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)
      print('Initialize ...')
      self.initialize(sess)

      print('Start profiling ...')
      for i in tqdm(range(0, timeline_skip + timeline_iter), ncols=80):
        if i < timeline_skip:
          summary, _ = sess.run([self.summ_train, self.train_op])
        else:
          summary, _ = sess.run([self.summ_train, self.train_op], 
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

  def param_stats(self):
    # build the computation graph
    self.build_train_graph()

    # get variables
    train_vars = tf.trainable_variables()

    # print
    total_num = 0
    for idx, v in enumerate(train_vars):
      shape = v.get_shape()
      shape_str = '; '.join([str(s) for s in shape])
      shape_num = shape.num_elements()
      print("{:3}, {:15}, [{}], {}".format(idx, v.name, shape_str, shape_num))
      total_num += shape_num
    print('Total trainable parameters: {}'.format(total_num))

  def test(self):
    # build graph
    self.build_test_graph()

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=10)

    # start
    num_tensors = len(self.test_tensors)
    avg_test = [0] * num_tensors
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)
      self.summ2txt(self.test_names, 'batch')

      # restore and initialize
      self.initialize(sess)
      print('Restore from checkpoint: %s' % self.flags.ckpt)
      tf_saver.restore(sess, self.flags.ckpt)

      print('Start testing ...')
      for i in range(0, self.flags.test_iter):
        iter_test_result = sess.run(self.test_tensors)
        iter_test_result = self.result_callback(iter_test_result)
        # run testing average
        for j in range(num_tensors):
          avg_test[j] += iter_test_result[j]
        # print the results
        reports = 'batch: %04d; ' % i
        for j in range(num_tensors):
          reports += '%s: %0.4f; ' % (self.test_names[j], iter_test_result[j])
        print(reports)
        self.summ2txt(iter_test_result, i)

    # Final testing results
    for j in range(num_tensors):
      avg_test[j] /= self.flags.test_iter
    avg_test = self.result_callback(avg_test)
    # print the results
    print('Testing done!\n')
    reports = 'ALL: %04d; ' % self.flags.test_iter
    for j in range(num_tensors):
      reports += '%s: %0.4f; ' % (self.test_names[j], avg_test[j])
    print(reports)
    self.summ2txt(avg_test, 'ALL')

  def run(self):
    eval('self.{}()'.format(self.flags.run))
