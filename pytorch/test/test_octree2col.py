import os
import torch
import ocnn
import unittest
import numpy as np
from torch.autograd import gradcheck


class Octree2ColTest(unittest.TestCase):
  # automatically call setUp() for every single test
  def setUp(self):  # def initialize(self):
    self.depth = 1
    self.channel = 3
    self.octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1']))
    self.data_in = np.random.uniform(-1.0, 1.0, [1, self.channel, 8, 1]).astype('float32')
    self.idx_maps = [list(range(0, 27)), [13],
                     [13, 14, 16, 17, 22, 23, 25, 26],
                     [4, 13, 22], [10, 13, 16], [12, 13, 14],
                     [1,  4,  7, 10, 13, 16, 19, 22, 25],
                     [3,  4,  5, 12, 13, 14, 21, 22, 23],
                     [9, 10, 11, 12, 13, 14, 15, 16, 17]]

  def forward(self, kernel_size, stride, idx_map):
    kernel = kernel_size[0] * kernel_size[1] * kernel_size[2]
    btm_h = 8
    top_h = 8 if stride == 1 else 1
    data_out = np.zeros([self.channel, kernel, top_h], dtype=np.float32)
    val_in = self.data_in.ravel()
    val_out = data_out.ravel()
    for c in range(0, self.channel):
      for k in range(0, kernel):
        for h in range(0, top_h):
          z = h & 1
          y = (h & 2) >> 1
          x = h >> 2

          kmap = idx_map[k]
          dz = kmap % 3
          t = kmap // 3
          dy = t % 3
          dx = t // 3

          z = z + dz - 1
          y = y + dy - 1
          x = x + dx - 1

          if -1 < x and x < 2 and -1 < y and y < 2 and -1 < z and z < 2:
            val_out[(c*kernel + k)*top_h + h] = val_in[c*btm_h + x*4 + y*2 + z]

    return data_out

  def test_forward(self):
    stride = [1, 2]
    vi = [0, 2, 3, 6, 1]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]

    for i in range(len(stride)):
      for j in range(len(vi)):
        out_gt = self.forward(kernel_size[j], stride[i], self.idx_maps[vi[j]])

        octree = self.octree.cuda()
        data_in = torch.from_numpy(self.data_in).cuda()
        data_out = ocnn.octree2col(data_in, octree,
                                   self.depth, kernel_size[j], stride[i], False)

        data_out = data_out.cpu().detach().numpy()
        self.assertTrue(np.array_equal(data_out, out_gt))

  def test_backward(self):
    stride = [1, 2]
    vi = [0, 2, 3, 6, 1]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]

    for i in range(len(stride)):
      for j in range(len(vi)):
        octree = self.octree.cuda()
        data_in = torch.from_numpy(self.data_in).cuda().requires_grad_()

        params = [data_in, octree, self.depth, kernel_size[j], stride[i], False]
        succ = gradcheck(ocnn.octree2col, params, eps=1.0)
        self.assertTrue(succ)

  def test_forwardP1(self):
    # depth = 1, this layer is all non-empty
    stride = [1, 2]
    vi = [0, 2, 3, 6, 1]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]
    for i in range(len(stride)):
      for j in range(len(vi)):
        octree = self.octree.cuda()
        data_in = torch.from_numpy(self.data_in).cuda()
        data_out = ocnn.nn.octree2col(data_in, octree,
                                      self.depth, kernel_size[j], stride[i], True)
        data_out = data_out.cpu().detach().numpy()

        out_gt = self.forward(kernel_size[j], stride[i], self.idx_maps[vi[j]])
        self.assertTrue(np.array_equal(data_out, out_gt))

  def test_octree2colP(self):
    depth = 4
    channel = 5
    stride = [1, 2]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]
    samples = ocnn.octree_samples(['octree_1', 'octree_2', 'octree_2', 'octree_1'])
    octree = ocnn.octree_batch(samples).cuda()
    node_num = ocnn.octree_property(octree, 'node_num', depth)
    data_in = torch.rand(1, channel, node_num.item(), 1).cuda()
    data_in = ocnn.octree_depad(data_in, octree, depth)
    data_in1 = data_in.clone().requires_grad_()
    data1 = ocnn.octree_pad(data_in1, octree, depth, 0)
    data_in2 = data_in.clone().requires_grad_()

    # octree2colP = octree2col + depad
    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        out1 = ocnn.octree2col(data1, octree, depth, kernel_size[j], stride[i], False)
        if stride[i] == 1:
          ks, height = out1.size(1), out1.size(2)
          out1 = out1.view(1, -1, height, 1)
          out1 = ocnn.octree_depad(out1, octree, depth)
          out1 = out1.view(channel, ks, -1)
        out2 = ocnn.octree2col(data_in2, octree, depth, kernel_size[j], stride[i], True)
        
        pesudo_grad = torch.rand(out1.shape, dtype=out1.dtype, device=out1.device)
        out1.backward(pesudo_grad, retain_graph=True)
        out2.backward(pesudo_grad, retain_graph=True)

        # check
        self.assertTrue(np.array_equal(out1.detach().cpu().numpy(), 
                                       out2.detach().cpu().numpy()))
        self.assertTrue(np.allclose(data_in1.grad.cpu().numpy(), 
                                    data_in2.grad.cpu().numpy()))


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
