import tensorflow as tf

from config import parse_args, FLAGS
from tfsolver import TFSolver
from network_factory import seg_network
from dataset import DatasetFactory
from ocnn import loss_functions_seg, build_solver, get_seg_label
from libs import points_property, octree_property, octree_decode_key

# Add config
FLAGS.LOSS.point_wise = True


# get the label and pts
def get_point_info(points, mask_ratio=0, mask=-1):
  with tf.name_scope('points_info'):
    pts   = points_property(points, property_name='xyz', channel=4)
    label = points_property(points, property_name='label', channel=1)
    label = tf.reshape(label, [-1])
    label_mask = label > mask  # mask out invalid points, -1
    if mask_ratio > 0:         # random drop some points to speed up training
      rnd_mask = tf.random.uniform(tf.shape(label_mask)) > mask_ratio
      label_mask = tf.logical_and(label_mask, rnd_mask)
    pts   = tf.boolean_mask(pts, label_mask)
    label = tf.boolean_mask(label, label_mask)
  return pts, label


# IoU
def tf_IoU_per_shape(pred, label, class_num, mask=-1):
  with tf.name_scope('IoU'):
    # Set mask to 0 to filter unlabeled points, whose label is 0
    label_mask = label > mask  # mask out label
    pred = tf.boolean_mask(pred, label_mask)
    label = tf.boolean_mask(label, label_mask)
    pred = tf.argmax(pred, axis=1, output_type=tf.int32)

    intsc, union = [None] * class_num, [None] * class_num
    for k in range(class_num):
      pk, lk = tf.equal(pred, k), tf.equal(label, k)
      intsc[k] = tf.reduce_sum(tf.cast(pk & lk, dtype=tf.float32))
      union[k] = tf.reduce_sum(tf.cast(pk | lk, dtype=tf.float32))
  return intsc, union


# define the graph
class ComputeGraphSeg:
  def __init__(self, flags):
    self.flags = flags

  def create_dataset(self, flags_data):
    return DatasetFactory(flags_data)(return_iter=True)

  def __call__(self, dataset='train', training=True, reuse=False, gpu_num=1):
    FLAGS = self.flags
    with tf.device('/cpu:0'):
      flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
      data_iter = self.create_dataset(flags_data)

    tower_tensors = []
    for i in range(gpu_num):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('device_%d' % i):
          octree, _, points = data_iter.get_next()
          pts, label = get_point_info(points, flags_data.mask_ratio)
          if not FLAGS.LOSS.point_wise:
            pts, label = None, get_seg_label(octree, FLAGS.MODEL.depth_out)
          logit = seg_network(octree, FLAGS.MODEL, training, reuse, pts=pts)
          losses = loss_functions_seg(logit, label, FLAGS.LOSS.num_class,
                                      FLAGS.LOSS.weight_decay, 'ocnn', mask=0)
          tensors = losses + [losses[0] + losses[2]]  # total loss
          names = ['loss', 'accu', 'regularizer', 'total_loss']

          if flags_data.batch_size == 1:
            num_class = FLAGS.LOSS.num_class
            intsc, union = tf_IoU_per_shape(logit, label, num_class, mask=0)
            iou = tf.constant(0.0)     # placeholder, calc its value later
            tensors = [iou] + tensors + intsc + union
            names = ['iou'] + names + \
                    ['intsc_%d' % i for i in range(num_class)] + \
                    ['union_%d' % i for i in range(num_class)]

          tower_tensors.append(tensors)
          reuse = True

    tensors = tower_tensors[0] if gpu_num == 1 else list(zip(*tower_tensors))
    return tensors, names


# define the solver
class PartNetSolver(TFSolver):
  def __init__(self, flags, compute_graph,  build_solver=build_solver):
    super(PartNetSolver, self).__init__(flags, compute_graph, build_solver)
    self.num_class = flags.LOSS.num_class # used to calculate the IoU

  def result_callback(self, avg_results):
    # calc part-IoU, update `iou`, this is in correspondence with Line 77
    iou_avg = 0.0
    ious = [0] * self.num_class
    for i in range(1, self.num_class):  # !!! Ignore the first label
      instc_i = avg_results[self.test_names.index('intsc_%d' % i)]
      union_i = avg_results[self.test_names.index('union_%d' % i)]
      ious[i] = instc_i / (union_i + 1.0e-10)
      iou_avg = iou_avg + ious[i]
    iou_avg = iou_avg / (self.num_class - 1)
    avg_results[self.test_names.index('iou')] = iou_avg
    return avg_results


# run the experiments
if __name__ == '__main__':
  FLAGS = parse_args()
  compute_graph = ComputeGraphSeg(FLAGS)
  solver = PartNetSolver(FLAGS, compute_graph)
  solver.run()
