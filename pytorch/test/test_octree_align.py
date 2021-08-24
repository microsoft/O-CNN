import os
import torch
import ocnn
import unittest
import numpy as np


class OctreeAlignTest(unittest.TestCase):
  def get_octree(self, filelist):
    batch = ocnn.octree_samples(filelist)
    return ocnn.octree_batch(batch).cuda()

  def test_forward_backward1(self):
    depth = 5
    octree = self.get_octree(['octree_1', 'octree_1'])
    data_in = torch.rand(1, 3, 16, 1).cuda().requires_grad_()
    data_out, idx = ocnn.octree_align(data_in, octree, octree, depth)
    idx_gt = torch.arange(16, dtype=torch.int32).cuda()

    out = data_out.sum()
    out.backward()
    grad_gt = np.ones([1, 3, 16, 1])

    self.assertTrue(np.array_equal(data_out.cpu().detach().numpy(),
                                   data_in.cpu().detach().numpy()))
    self.assertTrue(np.array_equal(idx.cpu().detach().numpy(),
                                   idx_gt.cpu().detach().numpy()))
    self.assertTrue(np.array_equal(data_in.grad.cpu().numpy(),
                                   grad_gt))

  def test_forward_backward2(self):
    depth = 5
    octree_in = self.get_octree(['octree_1'])
    octree_out = self.get_octree(['octree_1', 'octree_1'])

    data_in = torch.rand(1, 3, 8, 1).cuda().requires_grad_()
    data_out, idx = ocnn.octree_align(data_in, octree_in, octree_out, depth)
    zeros = torch.zeros(1, 3, 8, 1, dtype=torch.float32).cuda()
    data_gt = torch.cat([data_in, zeros], dim=2)
    idx_gt = torch.arange(8, dtype=torch.int32)

    out = data_out.sum()
    out.backward()
    grad_gt = np.ones([1, 3, 8, 1])

    self.assertTrue(np.array_equal(data_out.cpu().detach().numpy(),
                                   data_gt.cpu().detach().numpy()))
    self.assertTrue(np.array_equal(idx.cpu().detach().numpy(),
                                   idx_gt.cpu().detach().numpy()))
    self.assertTrue(np.array_equal(data_in.grad.cpu().numpy(),
                                   grad_gt))

  def test_forward_backward3(self):
    depth = 5
    octree_in = self.get_octree(['octree_1', 'octree_1'])
    octree_out = self.get_octree(['octree_1'])
    data_in = torch.rand(1, 3, 16, 1).cuda().requires_grad_()
    data_out, idx = ocnn.octree_align(data_in, octree_in, octree_out, depth)
    data_gt = data_in[:, :, :8, :]
    idx_gt = np.array(list(range(8)) + [-1] * 8)

    out = data_out.sum()
    out.backward()
    grad_gt = torch.cat([torch.ones(1, 3, 8, 1), torch.zeros(1, 3, 8, 1)], 2)

    self.assertTrue(np.array_equal(data_out.cpu().detach().numpy(),
                                   data_gt.cpu().detach().numpy()))
    self.assertTrue(np.array_equal(idx.cpu().detach().numpy(),
                                   idx_gt))
    self.assertTrue(np.array_equal(data_in.grad.cpu().numpy(),
                                   grad_gt))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
