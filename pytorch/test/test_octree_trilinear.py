import os
import torch
import ocnn
import unittest
import numpy as np


class OctreeTrilinearTest(unittest.TestCase):
  def test_forward1(self):
    depth, channel, nnum = 4, 3, 16
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_1'])).cuda()
    data = torch.ones([1, channel, nnum, 1], dtype=torch.float32).cuda()
    linear = ocnn.octree_trilinear(data, octree, depth, depth + 1)
    gt_result = np.ones([1, channel, 16, 1], dtype=np.float32)
    self.assertTrue((linear.cpu().numpy() == gt_result).all())

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
