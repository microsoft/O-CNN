import os
import sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
sys.path.append("../script")
from libs import *
from ocnn import octree2points

class Octree2PointsTest(tf.test.TestCase):

  def test_forward1(self):
    octree = octree_batch(octree_samples(['octree_3']))
    pts = octree2points(octree, depth=5, pts_channel=3)

    gt = np.array([[18.5, 16.605, 16.605], [18.5, 16.48, 17.5],
                   [18.5, 17.5, 16.48],    [18.5, 16.48, 18.5],
                   [18.5, 16.48, 19.5],    [18.5, 18.5, 16.48],
                   [18.5, 19.5, 16.48],    [16.48, 18.5, 20.5],
                   [18.5, 16.48, 20.5],    [18.5, 16.48, 21.5],
                   [16.48, 20.5, 18.5],    [18.5, 20.5, 16.48],
                   [18.5, 21.5, 16.48]])

    with self.cached_session() as sess:
      pt = sess.run(pts)
      self.assertAllClose(pt, gt)

  def test_forward2(self):
    octree = octree_batch(octree_samples(['octree_3', 'octree_3']))
    pts = octree2points(octree, depth=5, pts_channel=4)

    gt = np.array([[18.5, 16.605, 16.605], [18.5, 16.48, 17.5],
                   [18.5, 17.5, 16.48],    [18.5, 16.48, 18.5],
                   [18.5, 16.48, 19.5],    [18.5, 18.5, 16.48],
                   [18.5, 19.5, 16.48],    [16.48, 18.5, 20.5],
                   [18.5, 16.48, 20.5],    [18.5, 16.48, 21.5],
                   [16.48, 20.5, 18.5],    [18.5, 20.5, 16.48],
                   [18.5, 21.5, 16.48]])
    idx = np.concatenate([np.zeros([13, 1]), np.ones([13, 1])], axis=0)
    gt  = np.concatenate([np.concatenate([gt, gt], axis=0), idx], axis=1) 

    with self.cached_session() as sess:
      pt = sess.run(pts)
      self.assertAllClose(pt, gt)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()
