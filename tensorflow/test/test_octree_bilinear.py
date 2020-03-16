import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from libs import *


class OctreeBilinearTest(tf.test.TestCase):

  def test_forward1(self):
    depth  = 4
    channel= 3
    octree = octree_batch(octree_samples(['octree_1']))
    data = tf.ones([1, channel, 8, 1], dtype=tf.float32)

    bilinear_data = octree_bilinear_legacy(data, octree, depth, depth + 1)

    # test
    with self.cached_session() as sess:
      self.assertAllClose(bilinear_data, data)

  def test_forward2(self):
    depth, channel, nnum  = 4, 2, 16
    octree = octree_batch(octree_samples(['octree_1', 'octree_1']))
    data = tf.random.uniform([1, channel, nnum, 1], dtype=tf.float32)

    bilinear1 = octree_bilinear_legacy(data, octree, depth, depth + 1)    

    xyz5 = octree_property(octree, property_name='xyz', depth=depth+1, channel=1, dtype=tf.uint32)
    xyz5 = tf.reshape(xyz5, [-1])
    xyz5 = tf.cast(octree_decode_key(xyz5), dtype=tf.float32)
    # Attention: displacement 0.5, scale 0.5
    xyz5 += tf.constant([0.5, 0.5, 0.5, 0.0], dtype=tf.float32)
    xyz5 *= tf.constant([0.5, 0.5, 0.5, 1.0], dtype=tf.float32)
    bilinear2, _ = octree_bilinear_new(xyz5, data, octree, depth)

    # test
    with self.cached_session() as sess:
      b1, b2 = sess.run([bilinear1, bilinear2])
      self.assertAllClose(b1, b2)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()