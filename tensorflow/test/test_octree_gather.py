import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from libs import *


class OctreeGatherTest(tf.test.TestCase):
  def test_forward_backward(self):
    channel, height = 4, 5
    data = tf.random.uniform([1, channel, height, 1], dtype=tf.float32)
    index = np.array([0, 4, 2, -1, 2, 2, 4])

    out1 = tf.gather(data, index, axis=2)
    out2 = octree_gather(data, index)
    grad1 = tf.gradients(out1, data)
    grad2 = tf.gradients(out2, data)

    with self.cached_session() as sess:
      d, o1, o2, g1, g2 = sess.run([data, out1, out2, grad1, grad2])
      self.assertAllEqual(o1, o1)
      self.assertAllEqual(g1[0], g2[0])


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()