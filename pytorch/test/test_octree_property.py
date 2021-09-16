import os
import torch
import ocnn
import unittest
import numpy as np


class OctreePropertyTest(unittest.TestCase):
  def octree_property(self, on_cuda=True):
    batch_size = 2
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1'] * batch_size))
    if on_cuda:
      octree = octree.cuda()

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
    # test child from depth=0
    out = torch.cat([ocnn.octree_property(octree, 'child', d) for d in range(1, 6)])
    outs = ocnn.octree_property(octree, 'child')
    self.assertTrue(np.array_equal(outs[batch_size:].cpu().numpy(), out.cpu().numpy()))

    # test node number
    nnums = np.array([2, 16, 128, 16, 16, 16])
    nnum_cums = np.array([0, 2, 18, 146, 162, 178, 194])
    node_num = ocnn.octree_property(octree, 'node_num', 5)
    node_nums = ocnn.octree_property(octree, 'node_num')
    node_num_cum = ocnn.octree_property(octree, 'node_num_cum', 5)
    node_nums_cum = ocnn.octree_property(octree, 'node_num_cum')
    self.assertTrue(node_num.item() == nnums[5])
    self.assertTrue(node_num_cum.item() == nnum_cums[5])
    self.assertTrue(np.array_equal(node_nums.cpu().numpy(), nnums))
    self.assertTrue(np.array_equal(node_nums_cum.cpu().numpy(), nnum_cums))

    # test batch_size, depth, full_depth
    self.assertTrue(ocnn.octree_property(octree, 'batch_size').item() == batch_size)
    self.assertTrue(ocnn.octree_property(octree, 'depth').item() == 5)
    self.assertTrue(ocnn.octree_property(octree, 'full_depth').item() == 2)

    # TODO: test key, xyz, and label
    # out = ocnn.octree_property(octree, 'key', 5)

  def test_octree_property(self):
    self.octree_property(on_cuda=True)
    self.octree_property(on_cuda=False)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
