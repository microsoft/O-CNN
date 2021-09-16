import tensorflow as tf

from config import parse_args, FLAGS
from tfsolver import TFSolver
from network_factory import seg_network
from dataset import DatasetFactory
from ocnn import loss_functions_seg, tf_IoU_per_shape, get_seg_label
from libs import points_property

# Add config
FLAGS.LOSS.point_wise = True


# get the label and pts
def get_point_info(points, mask_ratio=0, mask=-1):
  with tf.name_scope('points_info'):    
    pts   = points_property(points, property_name='xyz', channel=4)
    label = points_property(points, property_name='label', channel=1)
    label = tf.reshape(label, [-1])
    label_mask  = label > mask # mask out invalid points, -1
    if mask_ratio > 0:         # random drop some points to speed up training
      rnd_mask = tf.random.uniform(tf.shape(label_mask)) > mask_ratio
      label_mask = tf.logical_and(label_mask, rnd_mask) 
    pts   = tf.boolean_mask(pts, label_mask)
    label = tf.boolean_mask(label, label_mask)
  return pts, label


class ComputeGraphSeg:
  def __init__(self, flags):
    self.flags = flags

  def __call__(self, dataset='train', training=True, reuse=False):
    FLAGS = self.flags
    flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
    batch = DatasetFactory(flags_data)()
    octree = batch[0]
    if FLAGS.LOSS.point_wise:
      pts, label = get_point_info(batch[2], flags_data.mask_ratio)
    else:
      pts, label = None, get_seg_label(octree, FLAGS.MODEL.depth_out)
    logit = seg_network(octree, FLAGS.MODEL, training, reuse, pts=pts)    
    losses = loss_functions_seg(logit, label, FLAGS.LOSS.num_class,
                                FLAGS.LOSS.weight_decay, 'ocnn', mask=-1)
    tensors = losses + [losses[0] + losses[2]] # total loss
    names = ['loss', 'accu', 'regularizer', 'total_loss']
    
    if flags_data.batch_size == 1:
      iou, valid_part_num = tf_IoU_per_shape(logit, label, FLAGS.LOSS.num_class)
      tensors += [iou, valid_part_num]
      names += ['iou', 'valid_part_num']
    return tensors, names


# run the experiments
if __name__ == '__main__':
  FLAGS = parse_args()
  compute_graph = ComputeGraphSeg(FLAGS)
  solver = TFSolver(FLAGS, compute_graph)
  solver.run()
