import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("..")
from libs import *

tf.enable_eager_execution()


class OctreeKeyTest(tf.test.TestCase):

  def test_decode_encode_key(self):
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    xyz = octree_property(octree, property_name='xyz', depth=5, channel=1, dtype=tf.uint64)
    xyz = tf.reshape(xyz, [-1])
    pts = octree_decode_key(xyz)
    xyz_encode = octree_encode_key(pts)

    gt = tf.constant([
        [16, 16, 16, 0], [16, 16, 17, 0], [16, 17, 16, 0], [16, 17, 17, 0],
        [17, 16, 16, 0], [17, 16, 17, 0], [17, 17, 16, 0], [17, 17, 17, 0],
        [16, 16, 16, 1], [16, 16, 17, 1], [16, 17, 16, 1], [16, 17, 17, 1],
        [17, 16, 16, 1], [17, 16, 17, 1], [17, 17, 16, 1], [17, 17, 17, 1]],
        dtype=tf.uint16)

    # test
    with self.cached_session() as sess:      
      self.assertAllEqual(gt, pts)
      self.assertAllEqual(xyz_encode, xyz)

  def test_xyz_key(self):
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    xyz = octree_property(octree, property_name='xyz', depth=5, channel=1, dtype=tf.uint64)
    xyz = tf.reshape(xyz, [-1])

    key = octree_xyz2key(xyz)
    xyz_out = octree_key2xyz(key)

    # test
    with self.cached_session() as sess:      
      self.assertAllEqual(xyz_out, xyz)

  def test_search_key(self):
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    key = tf.constant([28673, 281474976739335, 10], dtype=tf.uint64)
    idx_gt = tf.constant([1, 15, -1], dtype=tf.int32)
    idx = octree_search_key(key, octree, depth=5, is_xyz=False)

    # test
    with self.cached_session() as sess:      
      self.assertAllEqual(idx_gt, idx)



if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()
