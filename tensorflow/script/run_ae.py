import os
import sys
import shutil
import tensorflow as tf
from tqdm import tqdm
from config import FLAGS
from tfsolver import TFSolver
from dataset import DatasetFactory
from learning_rate import LRFactory
from network_ae import *
from ocnn import *


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if len(sys.argv) < 2:
  print('Usage: python run_cls.py config.ymal')

# update FLAGS
config_file = sys.argv[1]
FLAGS.merge_from_file(config_file)
FLAGS.freeze()

# backup the config file
if not os.path.exists(FLAGS.SOLVER.logdir):
  os.makedirs(FLAGS.SOLVER.logdir)
shutil.copy2(config_file, FLAGS.SOLVER.logdir)

# define the graph
def compute_graph(training=True, reuse=False):
  FLAGSD = FLAGS.DATA.train if training else FLAGS.DATA.test
  with tf.name_scope('dataset'):
    dataset = DatasetFactory(FLAGSD)
    octree, label = dataset()
  code = octree_encoder(octree, FLAGS.MODEL, training, reuse)
  loss, accu = octree_decoder(code, octree, FLAGS.MODEL, training, reuse)
  with tf.name_scope('compute_loss'):
    var_all = tf.trainable_variables()
    reg = tf.add_n([tf.nn.l2_loss(v) for v in var_all]) * FLAGS.LOSS.weight_decay
    total_loss  = tf.add_n(loss + [reg])
  tensors = loss + [reg] + accu + [total_loss]
  depth = FLAGS.MODEL.depth
  names = ['loss%d' % d for d in range(2, depth + 1)] + ['normal', 'reg'] + \
          ['accu%d' % d for d in range(2, depth + 1)] + ['total_loss']
  return tensors, names

# define the solver
class AeTFSolver(TFSolver):
  def __init__(self, flags):
    super(AeTFSolver, self).__init__(flags)

  def build_train_graph(self):
    self.train_tensors, tensor_names = compute_graph(training=True, reuse=False)
    self.test_tensors,  tensor_names = compute_graph(training=False, reuse=True)
    total_loss = self.train_tensors[-1]
    self.op_train, lr = build_solver(total_loss, LRFactory(self.flags))
    self.summaries(tensor_names + ['lr'], self.train_tensors + [lr,],
                   tensor_names)

  def build_test_graph(self):
    self.test_tensors, self.test_names = compute_graph(training=False, reuse=False)

  def decode_shape(self):
    # build graph
    FLAGSM = FLAGS.MODEL
    with tf.name_scope('dataset'):
        dataset = DatasetFactory(FLAGS.DATA.test)
        octree, label = dataset()
    code = octree_encoder(octree, FLAGSM, training=False, reuse=False)
    octree_pred = octree_decode_shape(code, FLAGSM, training=False, reuse=False)

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=20)

    # start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      # restore and initialize
      self.initialize(sess)
      tf_saver.restore(sess, self.flags.ckpt)
      logdir = self.flags.logdir
      tf.summary.FileWriter(logdir, sess.graph)

      print('Start testing ...')
      for i in tqdm(range(0, self.flags.test_iter)):
        origin, reconstructed = sess.run([octree, octree_pred])
        with open(logdir + ('/%04d_input.octree' % i), "wb") as f:
          f.write(origin.tobytes())
        with open(logdir + ('/%04d_output.octree' % i), "wb") as f:
          f.write(reconstructed.tobytes())

# run the experiments
solver = AeTFSolver(FLAGS.SOLVER)
if FLAGS.SOLVER.run == "train":
  solver.train()
elif FLAGS.SOLVER.run == "test":
  solver.test()
elif FLAGS.SOLVER.run == "decode_shape":
  solver.decode_shape()
else:
  print("Error! Unsupported FLAGS.SOLVER.run: " + FLAGS.SOLVER.run)
