import sys
import tensorflow as tf
sys.path.append("..")
from libs import bounding_sphere, points2octree, \
                 transform_points, octree_batch, normalize_points


class ParseExample:
  def __init__(self, x_alias='data', y_alias='label', **kwargs):
    self.x_alias = x_alias
    self.y_alias = y_alias
    self.features = { x_alias : tf.FixedLenFeature([], tf.string),
                      y_alias : tf.FixedLenFeature([], tf.int64) }

  def __call__(self, record):
    parsed = tf.parse_single_example(record, self.features)
    return parsed[self.x_alias], parsed[self.y_alias]


class Points2Octree:
  def __init__(self, depth, full_depth=2, node_dis=False, node_feat=False,
               split_label=False, adaptive=False, adp_depth=4, th_normal=0.1,
               save_pts=False, **kwargs):
    self.depth = depth
    self.full_depth = full_depth
    self.node_dis = node_dis
    self.node_feat = node_feat
    self.split_label = split_label
    self.adaptive = adaptive
    self.adp_depth = adp_depth
    self.th_normal = th_normal
    self.save_pts = save_pts

  def __call__(self, points):
    octree = points2octree(points, depth=self.depth, full_depth=self.full_depth,
                           node_dis=self.node_dis, node_feature=self.node_feat, 
                           split_label=self.split_label, adaptive=self.adaptive, 
                           adp_depth=self.adp_depth, th_normal=self.th_normal,
                           save_pts=self.save_pts)
    return octree

class NormalizePoints:
  def __init__(self):
    pass # normalize with bounding_sphere

  def __call__(self, points):
    radius, center = bounding_sphere(points)
    points = normalize_points(points, radius, center)
    return points


class TransformPoints:
  def __init__(self, distort, depth, offset=0.55, axis='xyz', scale=0.25, 
               jitter=0.25, drop_dim=[8, 32], angle=[20, 180, 20], dropout=[0, 0],
               stddev=[0, 0, 0], uniform=False, interval=[1, 1, 1], **kwargs):
    self.distort = distort
    self.axis = axis
    self.scale = scale
    self.jitter = jitter
    self.depth = depth
    self.offset = offset
    self.angle = angle
    self.drop_dim = drop_dim
    self.dropout = dropout
    self.stddev = stddev
    self.uniform_scale = uniform
    self.interval = interval

  def __call__(self, points):
    angle, scale, jitter, ratio, dim, angle, stddev = 0.0, 1.0, 0.0, 0.0, 0, 0, 0

    if self.distort:
      angle = [0, 0, 0]
      for i in range(3):
        interval = self.interval[i] if self.interval[i] > 1 else 1
        rot_num  = self.angle[i] // interval
        rnd = tf.random.uniform(shape=[], minval=-rot_num, maxval=rot_num, dtype=tf.int32)        
        angle[i] = tf.cast(rnd, dtype=tf.float32) * (interval * 3.14159265 / 180.0)
      angle = tf.stack(angle)

      minval, maxval = 1 - self.scale, 1 + self.scale
      scale = tf.random.uniform(shape=[3], minval=minval, maxval=maxval, dtype=tf.float32)
      if self.uniform_scale:
        scale = tf.stack([scale[0]]*3)
      
      minval, maxval = -self.jitter, self.jitter
      jitter = tf.random.uniform(shape=[3], minval=minval, maxval=maxval, dtype=tf.float32)
      
      minval, maxval = self.dropout[0], self.dropout[1]
      ratio = tf.random.uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.float32)
      minval, maxval = self.drop_dim[0], self.drop_dim[1]
      dim = tf.random.uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.int32)
      # dim = tf.cond(tf.random_uniform([], 0, 1) > 0.5, lambda: 0,
      #     lambda: tf.random.uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.int32))

      stddev = [tf.random.uniform(shape=[], minval=0, maxval=s) for s in self.stddev]
      stddev = tf.stack(stddev)

    radius, center = tf.constant(1.0), tf.constant([0.0, 0.0, 0.0])
    points = transform_points(points, angle=angle, scale=scale, jitter=jitter, 
                              radius=radius, center=center, axis=self.axis, 
                              depth=self.depth, offset=self.offset,
                              ratio=ratio, dim=dim, stddev=stddev)
    # The range of points is [-1, 1]
    return points # TODO: return the transformations


class PointDataset:
  def __init__(self, parse_example, normalize_points, transform_points, points2octree):
    self.parse_example = parse_example
    self.normalize_points = normalize_points
    self.transform_points = transform_points
    self.points2octree = points2octree

  def __call__(self, record_names, batch_size, shuffle_size=1000,
               return_iter=False, take=-1, return_pts=False, **kwargs):
    with tf.name_scope('points_dataset'):
      def preprocess(record):
        points, label = self.parse_example(record)
        points = self.normalize_points(points)
        points = self.transform_points(points)
        octree = self.points2octree(points)
        outputs= (octree, label)
        if return_pts: outputs += (points,)
        return outputs

      def merge_octrees(octrees, *args):
        octree = octree_batch(octrees)
        return (octree,) + args

      dataset = tf.data.TFRecordDataset(record_names).take(take).repeat()
      if shuffle_size > 1: dataset = dataset.shuffle(shuffle_size)
      itr = dataset.map(preprocess, num_parallel_calls=16) \
                   .batch(batch_size).map(merge_octrees, num_parallel_calls=8) \
                   .prefetch(8).make_one_shot_iterator() 
    return itr if return_iter else itr.get_next()


class OctreeDataset:
  def __init__(self, parse_example):
    self.parse_example = parse_example

  def __call__(self, record_names, batch_size, shuffle_size=1000,
               return_iter=False, take=-1, **kwargs):
    with tf.name_scope('octree_dataset'):
      def merge_octrees(octrees, labels):
        return octree_batch(octrees), labels

      dataset = tf.data.TFRecordDataset(record_names).take(take).repeat()
      if shuffle_size > 1: dataset = dataset.shuffle(shuffle_size)
      itr = dataset.map(self.parse_example, num_parallel_calls=8) \
                   .batch(batch_size).map(merge_octrees, num_parallel_calls=8) \
                   .prefetch(8).make_one_shot_iterator() 
    return itr if return_iter else itr.get_next()


class DatasetFactory:
  def __init__(self, flags, normalize_points=NormalizePoints,
               point_dataset=PointDataset, transform_points=TransformPoints):
    self.flags = flags
    if flags.dtype == 'points':
      self.dataset = point_dataset(ParseExample(**flags), normalize_points(),
                                   transform_points(**flags), Points2Octree(**flags))
    elif flags.dtype == 'octree':
      self.dataset = OctreeDataset(ParseExample(**flags))
    else:
      print('Error: unsupported datatype ' + flags.dtype)

  def __call__(self, return_iter=False):
    return self.dataset(
        record_names=self.flags.location, batch_size=self.flags.batch_size,
        shuffle_size=self.flags.shuffle, return_iter=return_iter,
        take=self.flags.take, return_pts=self.flags.return_pts)        
