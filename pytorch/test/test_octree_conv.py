import os
import torch
import ocnn
import unittest
import numpy as np


class OctreeConvTest(unittest.TestCase):

  def forward_and_backward(self, kernel_size, stride):
    depth = 4
    channel = 3
    height = 152
    num_outputs = 5
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_2']))
    data = np.random.uniform(-1.0, 1.0, [1, channel, height, 1]).astype('float32')

    # forward
    conv1 = ocnn.OctreeConv(depth, channel, num_outputs, kernel_size, stride)
    conv2 = ocnn.OctreeConvFast(depth, channel, num_outputs, kernel_size, stride)

    # use the same initialization
    with torch.no_grad():
      conv2.weights.data = conv1.weights.data

    # forward
    octree = octree.to('cuda')
    conv1.to('cuda')
    data1 = torch.from_numpy(data).to('cuda').requires_grad_()
    out1 = conv1(data1, octree)
    conv2.to('cuda')
    data2 = torch.from_numpy(data).to('cuda').requires_grad_()
    out2 = conv2(data2, octree)

    # backward
    pesudo_grad = torch.rand(out1.shape, dtype=out1.dtype, device=out1.device)
    out1.backward(pesudo_grad)
    out2.backward(pesudo_grad)

    # test
    self.assertTrue(np.array_equal(out1.cpu().detach().numpy(),
                                   out2.cpu().detach().numpy()))
    self.assertTrue(np.allclose(data1.grad.cpu().numpy(),
                                data2.grad.cpu().numpy(),
                                atol=1e-06))
    self.assertTrue(np.allclose(conv1.weights.grad.cpu().numpy(),
                                conv2.weights.grad.cpu().numpy(),
                                atol=1e-06))

  def test_forward_and_backward(self):
    stride = [1, 2]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]

    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        self.forward_and_backward(kernel_size[j], stride[i])


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
