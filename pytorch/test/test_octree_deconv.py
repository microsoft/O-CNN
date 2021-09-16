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
    num_outputs = 5
    octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_2']))
    data = np.random.uniform(-1.0, 1.0, [1, channel, height, 1]).astype('float32')

    # forward
    conv1 = ocnn.OctreeDeconv(depth, channel, num_outputs, kernel_size, stride)
    conv2 = ocnn.OctreeDeconvFast(depth, channel, num_outputs, kernel_size, stride)
    conv3 = ocnn.OctreeDeconv(depth, channel, num_outputs, kernel_size, stride, True)
    conv4 = ocnn.OctreeDeconv(depth, channel, num_outputs, kernel_size, stride)

    # use the same initialization
    with torch.no_grad():
      conv2.weights.data.copy_(conv1.weights.data)
      conv3.weights.data.copy_(conv1.weights.data)
      conv4.weights.data.copy_(conv1.weights.data)

    # forward - compare OctreeConv and OctreeConvFast
    octree = octree.cuda()
    conv1.cuda()
    data1 = torch.from_numpy(data).cuda().requires_grad_()
    out1 = conv1(data1, octree)

    conv2.cuda()
    data2 = torch.from_numpy(data).cuda().requires_grad_()
    out2 = conv2(data2, octree)

    # forward - compare OctreeConv with nempty = True and False
    conv3.cuda()
    mask3 = ocnn.octree_property(octree, 'child', depth) >= 0
    data3 = torch.from_numpy(data).cuda().requires_grad_()
    tmp3 = data3[:, :, mask3]
    out3 = conv3(tmp3, octree)

    conv4.cuda()
    depth_out = depth if stride == 1 else depth + 1
    mask4 = ocnn.octree_property(octree, 'child', depth_out) >= 0
    data4 = torch.from_numpy(data).cuda().requires_grad_()
    tmp4 = data4 * mask3.unsqueeze(-1).float()
    tmp4 = conv4(tmp4, octree)
    out4 = tmp4[:, :, mask4]

    # backward
    pesudo_grad = torch.rand(out1.shape, dtype=out1.dtype, device=out1.device)
    out1.backward(pesudo_grad)
    out2.backward(pesudo_grad)

    pesudo_grad2 = torch.rand(out3.shape, dtype=out3.dtype, device=out3.device)
    out3.backward(pesudo_grad2)
    out4.backward(pesudo_grad2)

    # test
    self.assertTrue(np.allclose(out1.cpu().detach().numpy(),
                                out2.cpu().detach().numpy(),
                                atol=1e-6))
    self.assertTrue(np.allclose(data1.grad.cpu().numpy(),
                                data2.grad.cpu().numpy(),
                                atol=1e-06))
    self.assertTrue(np.allclose(conv1.weights.grad.cpu().numpy(),
                                conv2.weights.grad.cpu().numpy(),
                                atol=1e-06))

    self.assertTrue(np.allclose(out3.cpu().detach().numpy(),
                                out4.cpu().detach().numpy(),
                                atol=1e-06))
    self.assertTrue(np.allclose(data3.grad.cpu().numpy(),
                                data4.grad.cpu().numpy(),
                                atol=1e-06))
    self.assertTrue(np.allclose(conv3.weights.grad.cpu().numpy(),
                                conv4.weights.grad.cpu().numpy(),
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
