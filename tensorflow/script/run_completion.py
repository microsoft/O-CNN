import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import parse_args, FLAGS
from tfsolver import TFSolver
from dataset import DatasetFactory
from network_completion import CompletionResnet
from ocnn import l2_regularizer
sys.path.append('..')
from libs import octree_scan, octree_batch, normalize_points


# flags
FLAGS.DATA.train.camera = '_' # used to generate partial scans
FLAGS.MODEL.skip_connections = True
FLAGS = parse_args()


# the dataset
class NormalizePoints:
  def __call__(self, points):
    radius = 64.0
    center = (64.0, 64.0, 64.0)
    points = normalize_points(points, radius, center)
    return points


class PointDataset:
  def __init__(self, parse_example, normalize_points, transform_points, points2octree):
    self.parse_example = parse_example
    self.normalize_points = normalize_points
    self.transform_points = transform_points
    self.points2octree = points2octree
    # reuse the DATA.train.camera for testing data
    with open(FLAGS.DATA.train.camera, 'rb') as fid:
      self.camera_path = pickle.load(fid)

  def gen_scan_axis(self, i):
    j = np.random.randint(0, 8)
    key = '%d_%d' % (i, j)
    axes = np.array(self.camera_path[key])
    # perturb the axes
    rnd = np.random.random(axes.shape) * 0.4 - 0.2
    axes = np.reshape(axes + rnd, (-1, 3))
    axes = axes / np.sqrt(np.sum(axes ** 2, axis=1, keepdims=True) + 1.0e-6)
    axes = np.reshape(axes, (-1))
    return axes.astype(np.float32)

  def __call__(self, record_names, batch_size, shuffle_size=1000,
               return_iter=False, take=-1, **kwargs):
    with tf.name_scope('points_dataset'):
      def preprocess(record):
        points, label = self.parse_example(record)
        points = self.normalize_points(points)
        points = self.transform_points(points)
        octree1 = self.points2octree(points)        # the complete octree
        scan_axis = tf.py_func(self.gen_scan_axis, [label], tf.float32)
        octree0 = octree_scan(octree1, scan_axis)   # the transformed octree
        return octree0, octree1

      def merge_octrees(octrees0, octrees1, *args):
        octree0 = octree_batch(octrees0)
        octree1 = octree_batch(octrees1)
        return (octree0, octree1) + args

      dataset = tf.data.TFRecordDataset(record_names).take(take).repeat()
      if shuffle_size > 1:
        dataset = dataset.shuffle(shuffle_size)
      itr = dataset.map(preprocess, num_parallel_calls=8) \
                   .batch(batch_size).map(merge_octrees, num_parallel_calls=8) \
                   .prefetch(8).make_one_shot_iterator()
    return itr if return_iter else itr.get_next()


# get the network
network = CompletionResnet(FLAGS.MODEL)


# define the graph
def compute_graph(dataset='train', training=True, reuse=False):
  flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
  octree0, octree1 = DatasetFactory(flags_data, NormalizePoints, PointDataset)()
  convd = network.octree_encoder(octree0, training, reuse)
  loss, accu = network.octree_decoder(convd, octree0, octree1, training, reuse)

  with tf.name_scope('total_loss'):
    reg = l2_regularizer('ocnn', FLAGS.LOSS.weight_decay)
    total_loss = tf.add_n(loss + [reg])
  tensors = loss + [reg] + accu + [total_loss]
  depth = FLAGS.MODEL.depth
  names = ['loss%d' % d for d in range(2, depth + 1)] + ['normal', 'reg'] + \
          ['accu%d' % d for d in range(2, depth + 1)] + ['total_loss']
  return tensors, names


# define the solver
class CompletionSolver(TFSolver):
  def __init__(self, flags, compute_graph):
    super(CompletionSolver, self).__init__(flags, compute_graph)

  def decode_shape(self):
    # build graph
    octree_in, _ = DatasetFactory(FLAGS.DATA.test, NormalizePoints)()
    convd = network.octree_encoder(octree_in, training=False, reuse=False)
    octree_out = network.decode_shape(convd, octree_in, training=False, reuse=False)

    # checkpoint
    assert(self.flags.ckpt)
    tf_saver = tf.train.Saver(max_to_keep=20)

    # start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      # restore and initialize
      self.initialize(sess)
      print('Load check point: ' + self.flags.ckpt)
      tf_saver.restore(sess, self.flags.ckpt)
      logdir = self.flags.logdir
      tf.summary.FileWriter(logdir, sess.graph)

      print('Start testing ...')
      for i in tqdm(range(0, self.flags.test_iter), ncols=80):
        o0, o1 = sess.run([octree_in, octree_out])
        with open(logdir + ('/%04d_input.octree' % i), "wb") as f:
          f.write(o0.tobytes())
        with open(logdir + ('/%04d_output.octree' % i), "wb") as f:
          f.write(o1.tobytes())


# run the experiments
if __name__ == '__main__':
  solver = CompletionSolver(FLAGS, compute_graph)
  solver.run()
