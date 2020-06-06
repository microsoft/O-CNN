import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append("..")
from libs import *


class OctreeAlignTest(tf.test.TestCase):
  def test_forward_backward1(self):
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    data_in = np.random.uniform(0.0, 1.0, size=[1, 3, 16, 1])
    data_in = tf.constant(data_in, dtype=tf.float32)
    data_out, idx = octree_align(data_in, octree, octree, depth=5)
    idx_gt = tf.range(16, dtype=tf.int32)

    grad = tf.gradients(data_out, data_in)[0]
    grad_gt = np.ones([1, 3, 16, 1])

    with self.cached_session() as sess:
      self.assertAllEqual(data_out, data_in)
      self.assertAllEqual(idx, idx_gt)
      self.assertAllEqual(grad, grad_gt)

  def test_forward_backward2(self):
    octree_in = octree_batch(octree_samples(['octree_1']))
    octree_out = octree_batch(octree_samples(['octree_1', 'octree_1']))
    data_in = np.random.uniform(0.0, 1.0, size=[1, 3, 8, 1])
    data_in = tf.constant(data_in, dtype=tf.float32)
    data_out, idx = octree_align(data_in, octree_in, octree_out, depth=5)
    data_gt = tf.concat([data_in, np.zeros([1, 3, 8, 1], np.float32)], axis=2)
    idx_gt = tf.range(8, dtype=tf.int32)

    grad = tf.gradients(data_out, data_in)[0]
    grad_gt = np.ones([1, 3, 8, 1])

    with self.cached_session() as sess:
      self.assertAllEqual(data_out, data_gt)
      self.assertAllEqual(idx, idx_gt)
      self.assertAllEqual(grad, grad_gt)

  def test_forward_backward3(self):
    octree_in = octree_batch(octree_samples(['octree_1', 'octree_1']))
    octree_out = octree_batch(octree_samples(['octree_1']))
    data_in = np.random.uniform(0.0, 1.0, size=[1, 3, 16, 1])
    data_in = tf.constant(data_in, dtype=tf.float32)
    data_out, idx = octree_align(data_in, octree_in, octree_out, depth=5)
    data_gt = data_in[:, :, :8, :]
    idx_gt = list(range(8)) + [-1] * 8

    grad = tf.gradients(data_out, data_in)[0]
    grad_gt = tf.concat([np.ones([1, 3, 8, 1]), np.zeros([1, 3, 8, 1])], axis=2)

    with self.cached_session() as sess:
      self.assertAllEqual(data_out, data_gt)
      self.assertAllEqual(idx, idx_gt)
      self.assertAllEqual(grad, grad_gt)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()
