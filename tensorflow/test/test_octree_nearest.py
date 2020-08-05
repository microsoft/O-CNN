import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from libs import *


class OctreeNearestTest(tf.test.TestCase):

  def test_forward_backward(self):
    depth, channel, nnum  = 4, 3, 16
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    data_shape = (1, channel, nnum, 1)
    data_np = np.random.uniform(-1, 1.0, data_shape).astype(np.float32)
    data = tf.constant(data_np)
    nearest1 = octree_tile(data, octree, depth)

    xyz = octree_xyz(octree, depth=depth+1, decode=False)
    xyz = tf.cast(octree_decode_key(xyz), dtype=tf.float32)    
    xyz += tf.constant([0.5, 0.5, 0.5, 0.0], dtype=tf.float32)
    xyz *= tf.constant([0.5, 0.5, 0.5, 1.0], dtype=tf.float32)    
    nearest2 = octree_nearest_interp(xyz, data, octree, depth)

    # test
    with self.cached_session() as sess:
      # forward
      b1, b2 = sess.run([nearest2, nearest1])
      self.assertAllClose(b1, b2)

      # backward
      grad_nn, grad_nm = tf.test.compute_gradient(
          data, data_shape, nearest2, b2.shape, delta=0.1)
      self.assertAllClose(grad_nn, grad_nm)  


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()