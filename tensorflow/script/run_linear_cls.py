import numpy as np
import tensorflow as tf

from config import parse_args
from tfsolver import TFSolver
from learning_rate import LRFactory
from ocnn import loss_functions, dense


FLAGS = parse_args()


class NumpyDataset:
  def __init__(self, flags):
    self.flags = flags
    self.data  = np.load('%s_%s.npy' % (flags.location, flags.x_alias))
    self.label = np.load('%s_%s.npy' % (flags.location, flags.y_alias))

  def __call__(self):
    with tf.name_scope('dataset'):
      channel = self.data.shape[1]
      self.data_ph  = tf.placeholder(dtype=tf.float32, shape=[None, channel])
      self.label_ph = tf.placeholder(dtype=tf.int64)
      dataset = tf.data.Dataset.from_tensor_slices((self.data_ph, self.label_ph))
      if self.flags.shuffle > 1:
        dataset = dataset.shuffle(self.flags.shuffle)
      dataset = dataset.batch(self.flags.batch_size).repeat()
      self.iter = dataset.make_initializable_iterator()
    return self.iter.get_next()

  def feed_data(self, sess):
    sess.run(self.iter.initializer, feed_dict = {self.data_ph : self.data,
                                                 self.label_ph: self.label})

# define the dataset
train_dataset = NumpyDataset(FLAGS.DATA.train)
test_dataset  = NumpyDataset(FLAGS.DATA.test)


# define the graph
def compute_graph(dataset='train', training=True, reuse=False):
  # get data
  numpy_dataset = train_dataset if dataset == 'train' else test_dataset
  data, label = numpy_dataset()
  # define the linear classifier
  with tf.variable_scope('linear', reuse=reuse):
    # TODO: Check that whether we need a BN here
    # data  = tf.layers.batch_normalization(data, axis=1, training=training)
    logit = dense(data, FLAGS.MODEL.nout, use_bias=True)
  # define the loss
  losses = loss_functions(logit, label, FLAGS.LOSS.num_class,
                          FLAGS.LOSS.weight_decay, 'linear')
  losses.append(losses[0] + losses[2]) # total loss
  names  = ['loss', 'accu', 'regularizer', 'total_loss']
  return losses, names


# define the solver
class LTFSolver(TFSolver):
  def initialize(self, sess):
    sess.run(tf.global_variables_initializer())
    train_dataset.feed_data(sess)
    test_dataset.feed_data(sess)


# run the experiments
solver = LTFSolver(FLAGS, compute_graph)
solver.run()
