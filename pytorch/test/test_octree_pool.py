import os
import torch
import ocnn
import unittest
import numpy as np


class OctreePoolTest(unittest.TestCase):

  def test_forward_and_backward_max_pool(self):
    depth, channel, height = 5, 2, 16
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_1']))
    data = np.array([[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
                     [8.1, 7.2, 6.3, 5.4, 4.5, 3.6, 2.7, 1.8]], dtype=np.float32)
    data = np.concatenate([data, data], axis=1)
    data = np.reshape(data, (1, channel, height, 1))
    out_gt = np.array([[8.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [8.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    out_gt = np.concatenate([out_gt, out_gt], axis=1)
    out_gt = np.reshape(out_gt, (1, channel, height, 1))
    grad_gt = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1],
                        [8.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    grad_gt = np.concatenate([grad_gt, grad_gt], axis=1)
    grad_gt = np.reshape(grad_gt, (1, channel, height, 1))
    mask_gt = np.array([[[[7], [15]], [[0], [8]]]], dtype=np.int32)

    # forward
    octree = octree.to('cuda')
    data_in = torch.from_numpy(data).to('cuda').requires_grad_()
    outputs, mask_out = ocnn.OctreeMaxPool(depth, return_indices=True)(data_in, octree)

    # backward
    pesudo_grad = torch.from_numpy(data).to('cuda')
    outputs.backward(pesudo_grad)

    # test
    self.assertTrue(np.array_equal(mask_out.cpu().detach().numpy(), mask_gt))
    self.assertTrue(np.array_equal(outputs.cpu().detach().numpy(), out_gt))
    self.assertTrue(np.array_equal(data_in.grad.cpu().numpy(), grad_gt))

  def test_forward_and_backward_max_unpool(self):
    depth, channel, height = 4, 2, 16
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_1']))
    data = np.array([[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
                     [8.1, 7.2, 6.3, 5.4, 4.5, 3.6, 2.7, 1.8]], dtype=np.float32)
    data = np.concatenate([data, data], axis=1)
    data = np.reshape(data, (1, channel, height, 1))
    mask = np.array([[[[1], [9]], [[2], [10]]]], dtype=np.int32)
    out_gt = np.array([[0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 8.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    out_gt = np.concatenate([out_gt, out_gt], axis=1)
    out_gt = np.reshape(out_gt, (1, channel, height, 1))
    grad_gt = np.array([[2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [6.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    grad_gt = np.concatenate([grad_gt, grad_gt], axis=1)
    grad_gt = np.reshape(grad_gt, (1, channel, height, 1))

    # forward
    octree = octree.to('cuda')
    mask = torch.from_numpy(mask).to('cuda')
    data_in = torch.from_numpy(data).to('cuda').requires_grad_()
    outputs = ocnn.OctreeMaxUnpool(depth)(data_in, mask, octree)

    # backward
    pesudo_grad = torch.from_numpy(data).to('cuda')
    outputs.backward(pesudo_grad)

    # test
    self.assertTrue(np.array_equal(outputs.cpu().detach().numpy(), out_gt))
    self.assertTrue(np.array_equal(data_in.grad.cpu().numpy(), grad_gt))


  def test_forward_and_backward_avg_pool(self):
    depth, channel, height = 5, 2, 16
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_1']))
    data = np.array([[8.0, 2.2, 3.3, 4.4, 5.5, 6.6, 7.6, 0.8],
                     [12.0,7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 8.0]], dtype=np.float32)
    data = np.concatenate([data, data], axis=1)
    data = np.reshape(data, (1, channel, height, 1))
    out_gt = np.array([[4.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    out_gt = np.concatenate([out_gt, out_gt], axis=1)
    out_gt = np.reshape(out_gt, (1, channel, height, 1))
    grad_gt = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]], dtype=np.float32)
    grad_gt = np.concatenate([grad_gt, grad_gt], axis=1)
    grad_gt = np.reshape(grad_gt, (1, channel, height, 1))
    mask_gt = np.array([[[[7], [15]], [[0], [8]]]], dtype=np.int32)

    # forward
    octree = octree.to('cuda')
    data_in = torch.from_numpy(data).to('cuda').requires_grad_()
    outputs = ocnn.OctreeAvgPool(depth)(data_in, octree)

    # backward
    pesudo_grad = torch.from_numpy(data).to('cuda')
    outputs.backward(pesudo_grad)

    # test
    self.assertTrue(np.array_equal(outputs.cpu().detach().numpy(), out_gt))
    self.assertTrue(np.array_equal(data_in.grad.cpu().numpy(), grad_gt))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
