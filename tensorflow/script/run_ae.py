import os
import tensorflow as tf
from tqdm import tqdm

from config import parse_args
from tfsolver import TFSolver
from dataset import DatasetFactory
from network_ae import make_autoencoder
from ocnn import l2_regularizer


FLAGS = parse_args()

# get the autoencoder
autoencoder = make_autoencoder(FLAGS.MODEL)

# define the graph
def compute_graph(dataset='train', training=True, reuse=False):
  flags_data = FLAGS.DATA.train if dataset=='train' else FLAGS.DATA.test
  octree, label = DatasetFactory(flags_data)()
  code = autoencoder.octree_encoder(octree, training, reuse)
  loss, accu = autoencoder.octree_decoder(code, octree, training, reuse)

  with tf.name_scope('total_loss'):
    reg = l2_regularizer('ocnn', FLAGS.LOSS.weight_decay)
    total_loss  = tf.add_n(loss + [reg])
  tensors = loss + [reg] + accu + [total_loss]
  depth = FLAGS.MODEL.depth
  names = ['loss%d' % d for d in range(2, depth + 1)] + ['normal', 'reg'] + \
          ['accu%d' % d for d in range(2, depth + 1)] + ['total_loss']
  return tensors, names

# define the solver
class AeTFSolver(TFSolver):

  def decode_shape(self):
    # build graph
    octree, label =  DatasetFactory(FLAGS.DATA.test)()
    code = autoencoder.octree_encoder(octree, training=False, reuse=False)
    octree_pred = autoencoder.octree_decode_shape(code, training=False, reuse=False)

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
if __name__ == '__main__':
  solver = AeTFSolver(FLAGS, compute_graph)
  solver.run()
