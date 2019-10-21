import sys
import tensorflow as tf
sys.path.append("..")
from libs import *


class PointsPreprocessor:
  def __init__(self, depth, distort, x_alias='octree', y_alias='label'):
    self._depth = depth
    self._distort = distort
    self._x_alias = x_alias
    self._y_alias = y_alias

  def __call__(self, record): 
    angle, scale, jitter, rot_num = 0.0, 1.0, 0.0, 24
    if self._distort:
      kPI = 2.0 * 3.14159265
      rd = tf.random.uniform(shape=[], minval=0, maxval=rot_num, dtype=tf.int32)
      angle = kPI * tf.cast(rd, dtype=tf.float32) / float(rot_num)
      scale = tf.random.uniform(shape=[], minval=0.75, maxval=1.25, dtype=tf.float32)
      jitter = tf.random.uniform(shape=[], minval=-2.0, maxval=2.0, dtype=tf.float32)

    points, label = self.parse_example(record)
    radius, center = bounding_sphere(points)
    points = transform_points(points, rotate=angle, scale=scale, jitter=jitter, 
                              radius=radius, center=center, axis='z', 
                              depth=self._depth, offset=0.55)
    octree = points2octree(points, depth=self._depth, full_depth=2, node_dis=False, 
                           split_label=False)
    return octree, label

  def parse_example(self, record):
    features = { self._x_alias : tf.FixedLenFeature([], tf.string),
                 self._y_alias : tf.FixedLenFeature([], tf.int64) }
    parsed = tf.parse_single_example(record, features)
    return parsed[self._x_alias], parsed[self._y_alias]


def octree_dataset(record_name, batch_size, x_alias='octree', y_alias='label'):
  def octree_record_parser(record):
    features = {x_alias: tf.FixedLenFeature([], tf.string),
                y_alias: tf.FixedLenFeature([], tf.int64)}
    parsed = tf.parse_example(record, features)
    octree = octree_batch(parsed[x_alias]) # merge_octrees
    label = parsed[y_alias]
    return octree, label

  with tf.name_scope('octree_dataset'):
    return tf.data.TFRecordDataset([record_name]).repeat().batch(batch_size)   \
                  .map(octree_record_parser, num_parallel_calls=8).prefetch(8) \
                  .make_one_shot_iterator().get_next()


def points_dataset(record_name, batch_size, depth=5, distort=False, 
                   x_alias='octree', y_alias='label'):
  def merge_octrees(octrees, labels):
    octree = octree_batch(octrees)
    return octree, labels

  with tf.name_scope('points_dataset'):
    return tf.data.TFRecordDataset([record_name]).repeat().shuffle(1000) \
                  .map(PointsPreprocessor(depth, distort, x_alias, y_alias), num_parallel_calls=8) \
                  .batch(batch_size).map(merge_octrees, num_parallel_calls=8) \
                  .prefetch(8).make_one_shot_iterator().get_next()