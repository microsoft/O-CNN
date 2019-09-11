import sys
import tensorflow as tf
sys.path.append("..")
from libs import *


class PointsPreprocessor:
  def __init__(self, distort, x_alias='octree', y_alias='label'):
    self._distort = distort
    self._x_alias = x_alias
    self._y_alias = y_alias

  def __call__(self, record): 
    angle = 0.0
    scale = 1.0
    jitter = 0.0
    if self._distort:
      rot_num = 24
      kPI = 2.0 * 3.14159265
      rd = tf.random.uniform(shape=[], minval=0, maxval=rot_num, dtype=tf.int32)
      angle = kPI * tf.cast(rd, dtype=tf.float32) / float(rot_num)
      scale = tf.random.uniform(shape=[], minval=0.75, maxval=1.25, dtype=tf.float32)
      jitter = tf.random.uniform(shape=[], minval=-2.0, maxval=2.0, dtype=tf.float32)

    points, label = self.parse_example(record)
    points = transform_points(points, rotate=angle, scale=scale, jitter=jitter, 
                              axis='z', depth=5, offset=0.55)
    octree = points2octree(points, depth=5, full_depth=2, node_dis=False, split_label=False)
    return octree, label

  def parse_example(self, record):
    features = { self._x_alias : tf.FixedLenFeature([], tf.string),
                 self._y_alias : tf.FixedLenFeature([], tf.int64) }
    parsed = tf.parse_single_example(record, features)
    return parsed[self._x_alias], parsed[self._y_alias]


def octree_dataset(record_name, batch_size):
  def octree_record_parser(record):
    features = {"octree": tf.FixedLenFeature([], tf.string),
                "label" : tf.FixedLenFeature([], tf.int64)}
    parsed = tf.parse_example(record, features)
    octree = octree_database(parsed["octree"]) # merge_octrees
    label = parsed["label"]
    return octree, label

  with tf.name_scope('octree_dataset'):
    return tf.data.TFRecordDataset([record_name]).repeat().batch(batch_size)   \
                  .map(octree_record_parser, num_parallel_calls=8).prefetch(8) \
                  .make_one_shot_iterator().get_next()


def points_dataset(record_name, batch_size, distort=False, x_alias='octree', y_alias='label'):
  def merge_octrees(octrees, labels):
    octree = octree_database(octrees)
    return octree, labels

  with tf.name_scope('points_dataset'):
    return tf.data.TFRecordDataset([record_name]).repeat().shuffle(1000) \
                  .map(PointsPreprocessor(distort, x_alias, y_alias), num_parallel_calls=8) \
                  .batch(batch_size).map(merge_octrees, num_parallel_calls=4)\
                  .prefetch(8).make_one_shot_iterator().get_next()