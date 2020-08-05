import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from libs import *


class OctreeSearchTest(tf.test.TestCase):

  def init_data(self):
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    data = np.array([[16.3, 16.5,  1.0, 16.2, 16.2, 16.0],
                     [16.3, 16.5,  1.0, 16.3, 16.3, 16.0],
                     [16.3, 16.5,  1.0, 17.1, 17.2, 15.0],
                     [ 0.0,  1.0,  0.0,  0.0,  1.0,  0.0]], dtype=np.float32)
    idx_gt  = np.array([0, 8, -1, 1, 9, -1], dtype=np.int32)
    return octree, data, idx_gt

  def test_forward1(self):
    octree, data, idx_gt = self.init_data()
    idx = octree_search(data, octree, depth=5)

    with self.cached_session() as sess:
      self.assertAllEqual(idx, idx_gt)

  def test_forward2(self):
    octree, data, idx_gt = self.init_data()

    data = tf.cast(data.T, tf_uints)
    data = octree_encode_key(data)
    idx = octree_search_key(data, octree, depth=5, is_xyz=True)
    with self.cached_session() as sess:
      self.assertAllEqual(idx, idx_gt)

  def test_forward3(self):
    octree, data, idx_gt = self.init_data()

    data = tf.cast(data.T, tf_uints)
    data = octree_encode_key(data)
    key = octree_xyz2key(data)
    idx = octree_search_key(key, octree, depth=5, is_xyz=False)
    with self.cached_session() as sess:
      self.assertAllEqual(idx, idx_gt)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()