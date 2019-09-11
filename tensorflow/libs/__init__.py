import os
import tensorflow as tf
from tensorflow.python.framework import ops

_current_path   = os.path.dirname(os.path.realpath(__file__))
_tf_ocnn_module = tf.load_op_library(os.path.join(_current_path, 'libocnn.so'))

octree_database = _tf_ocnn_module.octree_database
points_database = _tf_ocnn_module.points_database # todo: delete this operator
transform_points= _tf_ocnn_module.transform_points
points2octree   = _tf_ocnn_module.points_to_octree
octree_property = _tf_ocnn_module.octree_property
octree_pad      = _tf_ocnn_module.octree_pad
octree_depad    = _tf_ocnn_module.octree_depad
octree2col      = _tf_ocnn_module.octree_to_col
col2octree      = _tf_ocnn_module.col_to_octree
octree_grow     = _tf_ocnn_module.octree_grow
octree_new      = _tf_ocnn_module.octree_new
octree_update   = _tf_ocnn_module.octree_update
octree_align    = _tf_ocnn_module.octree_align
octree_mask     = _tf_ocnn_module.octree_mask
octree_set_property = _tf_ocnn_module.octree_set_property
_octree_max_pool    = _tf_ocnn_module.octree_max_pool
_octree_mask_pool   = _tf_ocnn_module.octree_mask_pool
_octree_max_unpool  = _tf_ocnn_module.octree_max_unpool
_octree_conv        = _tf_ocnn_module.octree_conv
_octree_deconv      = _tf_ocnn_module.octree_deconv
_octree_conv_grad   = _tf_ocnn_module.octree_conv_grad
_octree_deconv_grad = _tf_ocnn_module.octree_deconv_grad
_octree_align_grad  = _tf_ocnn_module.octree_align_grad


ops.NotDifferentiable("OctreeSetProperty")
ops.NotDifferentiable("OctreeDatabase")
ops.NotDifferentiable("PointsDatabase")
ops.NotDifferentiable("TransformPoints")
ops.NotDifferentiable("PointsToOctree")
ops.NotDifferentiable("OctreeProperty")
ops.NotDifferentiable("OctreeNew")
ops.NotDifferentiable("OctreeUpdate")
ops.NotDifferentiable("OctreeGrow")


@ops.RegisterGradient("OctreePad")
def _OctreePadGrad(op, grad):
  grad_out = octree_depad(grad, op.inputs[1], op.get_attr("depth"))
  return [grad_out, None]


@ops.RegisterGradient("OctreeDepad")
def _OctreeDepadGrad(op, grad):
  grad_out = octree_pad(grad, op.inputs[1], op.get_attr("depth"))
  return [grad_out, None]


@ops.RegisterGradient("OctreeToCol")
def _OctreeToColGrad(op, grad):
  grad_out = col2octree(grad, op.inputs[1], op.get_attr("depth"),
                        op.get_attr("kernel_size"), op.get_attr("stride"))
  return [grad_out, None]


@ops.RegisterGradient("ColToOctree")
def _ColToOctreeGrad(op, grad):
  grad_out = octree2col(grad, op.inputs[1], op.get_attr("depth"),
                        op.get_attr("kernel_size"), op.get_attr("stride"))
  return [grad_out, None]


@ops.RegisterGradient("OctreeMaxPool")
def _OctreeMaxPoolGrad(op, *grad):
  grad_out = _octree_max_unpool(grad[0], op.outputs[1], op.inputs[1], 
                                op.get_attr("depth"))
  return [grad_out, None]


@ops.RegisterGradient("OctreeMaxUnpool")
def _OctreeMaxUnpoolGrad(op, grad):
  grad_out = _octree_mask_pool(grad, op.inputs[1], op.inputs[2], 
                               op.get_attr("depth"))
  return [grad_out, None, None]


@ops.RegisterGradient("OctreeMaskPool")
def _OctreeMaskPoolGrad(op, grad):
  grad_out = _octree_max_unpool(grad, op.inputs[1], op.inputs[2], 
                               op.get_attr("depth"))
  return [grad_out, None, None]


@ops.RegisterGradient("OctreeConv")
def _OctreeConvGrad(op, grad):
  grad_out = _octree_conv_grad(op.inputs[0], op.inputs[1], op.inputs[2], grad,
                               op.get_attr("depth"), op.get_attr("num_output"),
                               op.get_attr("kernel_size"), op.get_attr("stride"))
  return grad_out + (None, )


@ops.RegisterGradient("OctreeDeconv")
def _OctreeDeconvGrad(op, grad):
  grad_out = _octree_deconv_grad(op.inputs[0], op.inputs[1], op.inputs[2], grad,
                                 op.get_attr("depth"), op.get_attr("num_output"),
                                 op.get_attr("kernel_size"), op.get_attr("stride"))
  return grad_out + (None, )


@ops.RegisterGradient("OctreeAlign")
def _OctreeAlignGrad(op, *grad):
  grad_out = octree_align_grad(grad[0], op.outputs[1])
  return [grad_out, None, None]


@ops.RegisterGradient("OctreeMask")
def _OctreeMaskGrad(op, grad):
  grad_out = octree_mask(grad, op.inputs[1], op.get_attr("mask"));
  return [grad_out, None]


def octree_max_pool(data, octree, depth):
  with tf.variable_scope("octree_max_pool"):
    data, mask = _octree_max_pool(data, octree, depth) # the bottom data depth
    data = octree_pad(data, octree, depth-1)           # !!! depth-1
  return data, mask


def octree_max_unpool(data, mask, octree, depth):
  with tf.variable_scope("octree_max_unpool"):  
    data = octree_depad(data, octree, depth)             # !!! depth
    data = _octree_max_unpool(data, mask, octree, depth) # the bottom data depth
  return data


def octree_conv_fast(data, octree, depth, channel):
  stride = 1       # todo: test stride == 2
  kernel_size = 3  # todo: test other kernel_size
  with tf.variable_scope("octree_conv"):      
    dim = int(data.shape[1]) * kernel_size**3
    kernel = tf.get_variable('weights', shape=[channel, dim],
        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    col = octree2col(data, octree, depth, kernel_size, stride)
    col = tf.reshape(col, [dim, -1])
    conv = tf.matmul(kernel, col)
  return tf.expand_dims(tf.expand_dims(conv, 0), -1)


def octree_conv_memory(data, octree, depth, channel):
  stride = 1       # todo: test stride == 2
  kernel_size = 3  # todo: test other kernel_size
  with tf.variable_scope("octree_conv"):      
    dim = int(data.shape[1]) * kernel_size**3
    kernel = tf.get_variable('weights', shape=[channel, dim],
        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    conv = _octree_conv(data, kernel, octree, depth, channel, kernel_size, stride)
    if stride == 2:
      conv = octree_pad(conv, octree, depth-1)
  return conv


def octree_deconv_memory(data, octree, depth, channel):
  stride = 1       # todo: test stride == 2
  kernel_size = 3  # todo: test other kernel_size
  with tf.variable_scope("octree_deconv"):      
    if stride == 2:
      conv = octree_depad(conv, octree, depth)
    dim = channel * kernel_size**3    
    kernel = tf.get_variable('weights', shape=[int(data.shape[1]), dim],
        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    deconv = _octree_deconv(data, kernel, octree, depth, channel, kernel_size, stride)
  return deconv


def octree_full_voxel(data, depth):
  height = 2 ** (3 * depth)
  channel = int(data.shape[1])
  with tf.variable_scope("octree_full_voxel"):
    data = tf.reshape(data, [channel, -1, height])
    data = tf.transpose(data, perm=[1, 0, 2])
  return data

