import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
# from octrees import *
from libs import *


class OctreePropertyTest(tf.test.TestCase):

  def test_octree_index(self):
    # octree_1 = get_one_octree('octree_1')
    # octree = octree_batch([octree_1, octree_1, octree_1])
    octree = octree_batch(octree_samples(['octree_1']*3))
    out = octree_property(octree, property_name='index', 
                          dtype=tf.int32, depth=5, channel=1)
    out = tf.reshape(out, [-1])
    out_gt = tf.constant([0]*8 + [1]*8 + [2]*8)

    # test
    with self.cached_session() as sess:  
      self.assertAllEqual(out, out_gt)

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()