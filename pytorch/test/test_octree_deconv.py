import os
import torch
import ocnn
import unittest
import numpy as np


class OctreeDeconvTest(unittest.TestCase):
  def forward_and_backward(self, kernel_size, stride, idx=0):
    depth = 4
    channel = 3
    height = 152
    num_outputs = 2
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_2']))
    data = np.random.uniform(-1.0, 1.0, [1, channel, height, 1]).astype('float32')

    # forward
    deconv1 = ocnn.OctreeDeconv(depth, channel, num_outputs, kernel_size, stride)
    deconv2 = ocnn.OctreeDeconvFast(depth, channel, num_outputs, kernel_size, stride)

    # use the same initialization
    with torch.no_grad():
      deconv2.weights.data = deconv1.weights.data

    # forward
    octree = octree.to('cuda')
    deconv1.to('cuda')
    data1 = torch.from_numpy(data).to('cuda').requires_grad_()
    out1 = deconv1(data1, octree)
    deconv2.to('cuda')
    data2 = torch.from_numpy(data).to('cuda').requires_grad_()
    out2 = deconv2(data2, octree)

    # backward
    pesudo_grad = torch.rand(out1.shape, dtype=out1.dtype, device=out1.device)
    out1.backward(pesudo_grad)
    out2.backward(pesudo_grad)

    # test
    self.assertTrue(np.allclose(out1.cpu().detach().numpy(),
                                out2.cpu().detach().numpy(),
                                atol=1e-6))
    self.assertTrue(np.allclose(data1.grad.cpu().numpy(),
                                data2.grad.cpu().numpy(),
                                atol=1e-06))
    self.assertTrue(np.allclose(deconv1.weights.grad.cpu().numpy(),
                                deconv2.weights.grad.cpu().numpy(),
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
