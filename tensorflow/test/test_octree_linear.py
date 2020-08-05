import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from libs import *


class OctreeLinearTest(tf.test.TestCase):

  def test_forward1(self):
    depth, channel = 4, 3
    octree = octree_batch(octree_samples(['octree_1']))
    data = tf.ones([1, channel, 8, 1], dtype=tf.float32)
    linear1 = octree_bilinear_legacy(data, octree, depth, depth + 1)
    linear2 = octree_bilinear(data, octree, depth, depth + 1)

    with self.cached_session() as sess:
      self.assertAllClose(linear1, data)
      self.assertAllClose(linear2, data)

  def test_forward_backward(self):
    depth, channel, nnum  = 4, 3, 16
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    data_shape = (1, channel, nnum, 1)
    data_np = np.random.uniform(-1, 1.0, data_shape).astype(np.float32)
    data = tf.constant(data_np)
    bilinear1 = octree_bilinear_legacy(data, octree, depth, depth + 1)    
    bilinear2 = octree_bilinear(data, octree, depth, depth + 1)

    # test
    with self.cached_session() as sess:
      # forward
      b1, b2 = sess.run([bilinear1, bilinear2])
      self.assertAllClose(b1, b2)

      # backward
      bilinear_shape = b2.shape
      grad_nn, grad_nm = tf.test.compute_gradient(
          data, data_shape, bilinear2, bilinear_shape, delta=0.1)
      self.assertAllClose(grad_nn, grad_nm)  


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()