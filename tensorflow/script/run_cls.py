import os
import sys
import shutil
import tensorflow as tf
from config import FLAGS
from tfsolver import TFSolver
from network_cls import network
from dataset import DatasetFactory
from learning_rate import LRFactory
from ocnn import *

# from network_hrnet_v1 import *
# network = network_cls

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
def compute_graph(dataset='train', training=True, reuse=False):
  FLAGSD = FLAGS.DATA.train if dataset=='train' else FLAGS.DATA.test
  with tf.name_scope('dataset'):
    dataset = DatasetFactory(FLAGSD)
    octree, label = dataset()
  logit = network(octree, FLAGS.MODEL, training, reuse)
  losses = loss_functions(logit, label, FLAGS.LOSS.num_class, 
                          FLAGS.LOSS.weight_decay, 'ocnn')
  return losses # loss, accu, regularizer

# define the solver
class ClsTFSolver(TFSolver):
  def __init__(self, flags):
    super(ClsTFSolver, self).__init__(flags)

  def build_train_graph(self):
    self.train_tensors = compute_graph(dataset='train', training=True, reuse=False)
    self.test_tensors  = compute_graph(dataset='test', training=False, reuse=True)
    total_loss = self.train_tensors[0] + self.train_tensors[2]
    self.op_train, lr = build_solver(total_loss, LRFactory(self.flags))
    
    tensor_names = ['loss', 'accu', 'regularizer']
    self.summaries(tensor_names + ['lr'], self.train_tensors + [lr,],
                   tensor_names)

  def build_test_graph(self):
    self.test_tensors = compute_graph(dataset='test', training=False, reuse=False)
    self.test_names = ['loss', 'accu', 'regularizer']


# run the experiments
solver = ClsTFSolver(FLAGS.SOLVER)
if FLAGS.SOLVER.run == "train":
  solver.train()
else:
  solver.test()
