import os
import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
# from octrees import *
from libs import *


class OctreeConvTest(tf.test.TestCase):

  def forward_and_backward(self, kernel_size, stride, idx=0):
    depth  = 4
    channel= 3
    height = 152
    num_outputs = 5
    # octree = octree_batch([get_one_octree('octree_1'), get_one_octree('octree_2')])
    octree = octree_batch(octree_samples(['octree_1', 'octree_2']))
    data = tf.constant(np.random.uniform(-1.0, 1.0, [1, channel, height, 1]).astype('float32'))

    # forward
    with tf.variable_scope('conv_%d' % idx) as scope:
      conv_fast = octree_conv_fast(data, octree, depth, num_outputs, kernel_size, stride)
      scope.reuse_variables()
      conv_mem = octree_conv_memory(data, octree, depth, num_outputs, kernel_size, stride)
    
    # get kernel
    t_vars = tf.trainable_variables()
    for var in t_vars:
      if ('conv_%d' % idx) in var.name:
        kernel = var

    # backward
    grad_fast, kernel_fast = tf.gradients(conv_fast, [data, kernel])
    grad_mem,  kernel_mem  = tf.gradients(conv_mem,  [data, kernel])   

    # test
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      # print('stride: ', stride, ', kernel_size: ', kernel_size)

      self.assertAllEqual(conv_fast, conv_mem)
      self.assertAllClose(grad_fast, grad_mem)
      self.assertAllClose(kernel_fast, kernel_mem)

  def test_forward_and_backward(self):
    idx = 0
    stride = [1, 2]
    kernel_size = [[3, 3, 3], [2, 2, 2], [3, 1, 1], [3, 3, 1], [1, 1, 1]]

    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        self.forward_and_backward(kernel_size[j], stride[i], idx)
        idx += 1


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()