import os
import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf
# from octrees import *
from libs import *


class Octree2ColTest(tf.test.TestCase):
  def initialize(self):
    self.depth  = 1
    self.channel= 3
    # self.octree = octree_batch(get_one_octree('octree_1'))
    self.octree = octree_batch(octree_samples(['octree_1']))
    self.data_in = np.random.uniform(-1.0, 1.0, [1, self.channel, 8, 1]).astype('float32')
    self.idx_maps= [list(range(0, 27)), [13], 
                    [13, 14, 16, 17, 22, 23, 25, 26],
                    [4, 13, 22], [10, 13, 16], [12, 13, 14],
                    [1,  4,  7, 10, 13, 16, 19, 22, 25],
                    [3,  4,  5, 12, 13, 14, 21, 22, 23],
                    [9, 10, 11, 12, 13, 14, 15, 16, 17]]

  def forward(self, kernel_size, stride, idx_map):
    kernel = kernel_size[0] * kernel_size[1] * kernel_size[2]
    # kernel = 27
    btm_h = 8
    top_h  = 8 if stride == 1 else 1
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
            t  = kmap // 3
            dy = t % 3
            dx = t // 3

            z = z + dz - 1
            y = y + dy - 1
            x = x + dx - 1

            if -1 < x and x < 2 and -1 < y and y < 2 and -1 < z and z < 2:
              val_out[(c*kernel + k)*top_h + h] = val_in[c*btm_h + x*4 + y*2 + z]

    return data_out


  def test_forward(self):
    self.initialize()
    stride = [1, 2]
    vi = [0, 2, 3, 6, 1]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]

    for i in range(len(stride)):
      for j in range(len(vi)):
        data_out = octree2col(self.data_in, self.octree, depth=self.depth, 
                              kernel_size=kernel_size[j], stride=stride[i])
        with self.cached_session():
          out_nn = data_out.eval()
          out_gt = self.forward(kernel_size[j], stride[i], self.idx_maps[vi[j]])
          self.assertAllEqual(out_nn, out_gt, 'forward: i=%d, j=%d' % (i, j))
  

  def test_backward(self):
    self.initialize()
    stride = [1, 2]
    vi = [0, 2, 3, 6, 1]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]

    for i in range(len(stride)):
      for j in range(len(vi)):
        data_in = tf.constant(self.data_in)
        data_out = octree2col(data_in, self.octree, depth=self.depth, 
                              kernel_size=kernel_size[j], stride=stride[i])
        with self.cached_session():
          out_nn = data_out.eval()
          shape_in = self.data_in.shape
          shape_out = out_nn.shape
          grad_nn, grad_nm = tf.test.compute_gradient(data_in, shape_in, 
              data_out, shape_out, delta=0.1)
          self.assertAllClose(grad_nn, grad_nm, msg='backward: i=%d, j=%d' % (i, j))  


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()