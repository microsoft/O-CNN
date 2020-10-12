import os
import torch
import ocnn
import unittest
import numpy as np


class OctreePropertyTest(unittest.TestCase):

  def test_octree_property(self):
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1'] * 2)).cuda()

    # test index
    out = ocnn.octree_property(octree, 'index', 5)
    out_gt = np.array([0] * 8 + [1] * 8)
    self.assertTrue(np.array_equal(out.cpu().numpy(), out_gt))

    # test feature
    out = ocnn.octree_property(octree, 'feature', 5)
    out_gt = np.zeros([3, 16], dtype=np.float32)
    out_gt[:, 0] = 3.0 ** 0.5 / 3.0
    out_gt[:, 8] = 3.0 ** 0.5 / 3.0
    out_gt = np.expand_dims(out_gt, axis=[0, 3])
    self.assertTrue(np.allclose(out.cpu().numpy(), out_gt))

    # test child
    out = ocnn.octree_property(octree, 'child', 5)
    out_gt = np.ones(16) * (-1)
    out_gt[0] = 0
    out_gt[8] = 1
    self.assertTrue(np.array_equal(out.cpu().numpy(), out_gt))

    # TODO: test key, xyz, and label
    # out = ocnn.octree_property(octree, 'key', 5)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
