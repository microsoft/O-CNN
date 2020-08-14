import tensorflow as tf
from config import parse_args
from tfsolver import TFSolver
from dataset import DatasetFactory
from learning_rate import LRFactory
from network_hrnet import HRNet
from mid_loss import ShapeLoss, PointLoss
from ocnn import l2_regularizer, build_solver, get_seg_label
from libs import octree_property



# configs
FLAGS = parse_args()


# get the label and mask
def get_point_info(octree, depth, mask_ratio=0):
  with tf.name_scope('points_info'):
    point_id = get_seg_label(octree, depth)
    point_segment = tf.reshape(octree_property(octree, property_name='index',
                             dtype=tf.int32, depth=depth, channel=1), [-1])
    mask = point_id > -1   # Filter out label -1
    if mask_ratio > 0:
      mask_shape = tf.shape(mask)
      mask = tf.logical_and(mask, tf.random.uniform(mask_shape) > mask_ratio)
    point_id = tf.boolean_mask(point_id, mask)
    point_segment = tf.boolean_mask(point_segment, mask)
  return point_id, point_segment, mask


def compute_graph(reuse=False):
  with tf.name_scope('dataset'):
    flags_data = FLAGS.DATA.train
    batch_size = flags_data.batch_size
    octree, shape_id = DatasetFactory(flags_data)()
    point_id, point_segment, point_mask = get_point_info(
        octree, FLAGS.MODEL.depth_out, flags_data.mask_ratio)

  # build model
  hrnet = HRNet(FLAGS.MODEL)
  tensors = hrnet.network(octree, training=True, reuse=reuse, mask=point_mask)
  shape_feature, point_feature = tensors['logit_cls'], tensors['logit_seg']

  # shape-level discrimination
  shape_critic = ShapeLoss(FLAGS.LOSS, reuse=reuse)
  shape_logit = shape_critic.forward(shape_feature)
  shape_loss, shape_accu = shape_critic.loss(shape_logit, shape_id)

  # point-level discrimination
  point_critic = PointLoss(FLAGS.LOSS, reuse=reuse)
  point_logit = point_critic.forward(point_feature, shape_id, point_segment, batch_size)
  point_loss, point_accu = point_critic.loss(point_logit, point_id)

  # run SGD
  reg = l2_regularizer('ocnn', FLAGS.LOSS.weight_decay)
  weights = FLAGS.LOSS.weights
  total_loss = shape_loss * weights[0] + point_loss * weights[1] + reg
  solver, lr = build_solver(total_loss, LRFactory(FLAGS.SOLVER))

  # update memory
  shape_update = shape_critic.update_memory(solver)
  point_update = point_critic.update_memory(solver)
  train_op = tf.group([shape_update, point_update, solver])

  return shape_loss, shape_accu, point_loss, point_accu, reg, lr, train_op


# define the solver
class MidTFSolver(TFSolver):
  def __init__(self, flags):
    super(MidTFSolver, self).__init__(flags)

  def build_train_graph(self):
    tensors = compute_graph(reuse=False)
    self.train_tensors = tensors[:-1]
    self.train_op = tensors[-1]
    self.test_tensors  = [tf.constant(0)]
    names = ['shape_loss', 'shape_accu', 'point_loss', 'point_accu', 'reg', 'lr']
    self.summaries(names, self.train_tensors, ['no_test_loss'])


# run the experiments
solver = MidTFSolver(FLAGS)
solver.train()
