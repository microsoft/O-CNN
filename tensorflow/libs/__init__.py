import os
import tensorflow as tf
from tensorflow.python.framework import ops

if 'OCTREE_KEY' in os.environ and os.environ['OCTREE_KEY'] == '64':
  print('INFO from ocnn: The octree key is 64 bits')
  octree_key64 = True
  tf_uintk = tf.uint64
  tf_uints = tf.uint16
  tf_intk = tf.int64
else:
  print('INFO from ocnn: The octree key is 32 bits, '
        'the octree depth should be smaller than 8. ')
  octree_key64 = False
  tf_uintk = tf.uint32
  tf_uints = tf.uint8
  tf_intk = tf.int32

_current_path = os.path.dirname(os.path.realpath(__file__))
_tf_ocnn_module = tf.load_op_library(os.path.join(_current_path, 'libocnn.so'))

bounding_sphere     = _tf_ocnn_module.bounding_sphere
points_property     = _tf_ocnn_module.points_property
transform_points    = _tf_ocnn_module.transform_points
normalize_points    = _tf_ocnn_module.normalize_points
points_new          = _tf_ocnn_module.points_new
points_set_property = _tf_ocnn_module.points_set_property
octree_drop         = _tf_ocnn_module.octree_drop
octree_scan         = _tf_ocnn_module.octree_scan
octree_cast         = _tf_ocnn_module.octree_cast
octree_batch        = _tf_ocnn_module.octree_batch
points2octree       = _tf_ocnn_module.points_to_octree
octree_property     = _tf_ocnn_module.octree_property
octree_pad          = _tf_ocnn_module.octree_pad
octree_depad        = _tf_ocnn_module.octree_depad
octree2col          = _tf_ocnn_module.octree_to_col
col2octree          = _tf_ocnn_module.col_to_octree
octree_grow         = _tf_ocnn_module.octree_grow
octree_new          = _tf_ocnn_module.octree_new
octree_update       = _tf_ocnn_module.octree_update
octree_align        = _tf_ocnn_module.octree_align
octree_mask         = _tf_ocnn_module.octree_mask
octree_samples      = _tf_ocnn_module.octree_samples
octree_search       = _tf_ocnn_module.octree_search
octree_key2xyz      = _tf_ocnn_module.octree_key_to_xyz
octree_xyz2key      = _tf_ocnn_module.octree_xyz_to_key
octree_decode_key   = _tf_ocnn_module.octree_decode_key
octree_encode_key   = _tf_ocnn_module.octree_encode_key
octree_search_key   = _tf_ocnn_module.octree_search_key
octree_set_property = _tf_ocnn_module.octree_set_property
octree_gather       = _tf_ocnn_module.octree_gather
octree_gatherbk     = _tf_ocnn_module.octree_gatherbk
_octree_max_pool    = _tf_ocnn_module.octree_max_pool
_octree_mask_pool   = _tf_ocnn_module.octree_mask_pool
_octree_max_unpool  = _tf_ocnn_module.octree_max_unpool
_octree_conv        = _tf_ocnn_module.octree_conv
_octree_deconv      = _tf_ocnn_module.octree_deconv
_octree_conv_grad   = _tf_ocnn_module.octree_conv_grad
_octree_deconv_grad = _tf_ocnn_module.octree_deconv_grad
_octree_align_grad  = _tf_ocnn_module.octree_align_grad
_octree_bilinear    = _tf_ocnn_module.octree_bilinear


ops.NotDifferentiable('BoundingSphere')
ops.NotDifferentiable('OctreeSetProperty')
ops.NotDifferentiable('OctreeBatch')
ops.NotDifferentiable('TransformPoints')
ops.NotDifferentiable('NormalizePoints')
ops.NotDifferentiable('PointsNew')
ops.NotDifferentiable('PointsSetProperty')
ops.NotDifferentiable('PointsToOctree')
ops.NotDifferentiable('OctreeProperty')
ops.NotDifferentiable('OctreeNew')
ops.NotDifferentiable('OctreeUpdate')
ops.NotDifferentiable('OctreeGrow')
ops.NotDifferentiable('OctreeSamples')
ops.NotDifferentiable('OctreeBilinear')
ops.NotDifferentiable('OctreeKeyToXyz')
ops.NotDifferentiable('OctreeXyzToKey')
ops.NotDifferentiable('OctreeDecodeKey')
ops.NotDifferentiable('OctreeEncodeKey')
ops.NotDifferentiable('OctreeSearchKey')
ops.NotDifferentiable('OctreeSearch')
ops.NotDifferentiable('PointsProperty')
ops.NotDifferentiable('OctreeScan')
ops.NotDifferentiable('OctreeCast')
ops.NotDifferentiable('OctreeDrop')


@ops.RegisterGradient('OctreePad')
def _OctreePadGrad(op, grad):
  grad_out = octree_depad(grad, op.inputs[1], op.get_attr('depth'))
  return [grad_out, None]


@ops.RegisterGradient('OctreeDepad')
def _OctreeDepadGrad(op, grad):
  grad_out = octree_pad(grad, op.inputs[1], op.get_attr('depth'))
  return [grad_out, None]


@ops.RegisterGradient('OctreeToCol')
def _OctreeToColGrad(op, grad):
  grad_out = col2octree(grad, op.inputs[1], op.get_attr('depth'),
                        op.get_attr('kernel_size'), op.get_attr('stride'))
  return [grad_out, None]


@ops.RegisterGradient('ColToOctree')
def _ColToOctreeGrad(op, grad):
  grad_out = octree2col(grad, op.inputs[1], op.get_attr('depth'),
                        op.get_attr('kernel_size'), op.get_attr('stride'))
  return [grad_out, None]


@ops.RegisterGradient('OctreeMaxPool')
def _OctreeMaxPoolGrad(op, *grad):
  grad_out = _octree_max_unpool(grad[0], op.outputs[1], op.inputs[1],
                                op.get_attr('depth'))
  return [grad_out, None]


@ops.RegisterGradient('OctreeMaxUnpool')
def _OctreeMaxUnpoolGrad(op, grad):
  grad_out = _octree_mask_pool(grad, op.inputs[1], op.inputs[2],
                               op.get_attr('depth'))
  return [grad_out, None, None]


@ops.RegisterGradient('OctreeMaskPool')
def _OctreeMaskPoolGrad(op, grad):
  grad_out = _octree_max_unpool(grad, op.inputs[1], op.inputs[2],
                                op.get_attr('depth'))
  return [grad_out, None, None]


@ops.RegisterGradient('OctreeConv')
def _OctreeConvGrad(op, grad):
  grad_out = _octree_conv_grad(op.inputs[0], op.inputs[1], op.inputs[2], grad,
                               op.get_attr('depth'), op.get_attr('num_output'),
                               op.get_attr('kernel_size'), op.get_attr('stride'))
  return grad_out + (None, )


@ops.RegisterGradient('OctreeDeconv')
def _OctreeDeconvGrad(op, grad):
  grad_out = _octree_deconv_grad(op.inputs[0], op.inputs[1], op.inputs[2], grad,
                                 op.get_attr('depth'), op.get_attr('num_output'),
      op.get_attr('kernel_size'), op.get_attr('stride'))
  return grad_out + (None, )


@ops.RegisterGradient('OctreeAlign')
def _OctreeAlignGrad(op, *grad):
  grad_out = _octree_align_grad(grad[0], op.outputs[1])
  return [grad_out, None, None]


@ops.RegisterGradient('OctreeMask')
def _OctreeMaskGrad(op, grad):
  grad_out = octree_mask(grad, op.inputs[1], op.get_attr('mask'))
  return [grad_out, None]


@ops.RegisterGradient('OctreeGather')
def _OctreeGatherGrad(op, grad):
  shape = tf.shape(op.inputs[0])
  grad_out = octree_gatherbk(grad, op.inputs[1], shape)
  return [grad_out, None]


def octree_max_pool(data, octree, depth):
  with tf.variable_scope('octree_max_pool'):
    data, mask = _octree_max_pool(data, octree, depth) # the bottom data depth
    data = octree_pad(data, octree, depth-1)           # !!! depth-1
  return data, mask


def octree_max_unpool(data, mask, octree, depth):
  with tf.variable_scope('octree_max_unpool'):
    data = octree_depad(data, octree, depth)                 # !!! depth
    data = _octree_max_unpool(data, mask, octree, depth + 1) # the bottom data depth
  return data


def octree_avg_pool(data, octree, depth):
  with tf.variable_scope('octree_avg_pool'):
    data = tf.reshape(data, [1, int(data.shape[1]), -1, 8])
    data = tf.reduce_mean(data, axis=3, keepdims=True)
    data = octree_pad(data, octree, depth-1)           # !!! depth-1
  return data


# todo: merge octree_conv_fast and octree_conv_memory to reduce code redundancy
def octree_conv_fast(data, octree, depth, channel, kernel_size=[3], stride=1):
  assert(type(kernel_size) is list and len(kernel_size) < 4)
  for i in range(len(kernel_size), 3):
    kernel_size.append(kernel_size[-1])

  with tf.variable_scope('octree_conv'):
    dim = int(data.shape[1]) * kernel_size[0] * kernel_size[1] * kernel_size[2]
    kernel = tf.get_variable('weights', shape=[channel, dim], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    col = octree2col(data, octree, depth, kernel_size, stride)
    col = tf.reshape(col, [dim, -1])
    conv = tf.matmul(kernel, col)
    conv = tf.expand_dims(tf.expand_dims(conv, 0), -1) # [C, H] -> [1, C, H, 1]
    if stride == 2:
      conv = octree_pad(conv, octree, depth-1, 0)
  return conv


def octree_conv_memory(data, octree, depth, channel, kernel_size=[3], stride=1):
  assert(type(kernel_size) is list and len(kernel_size) < 4)
  for i in range(len(kernel_size), 3):
    kernel_size.append(kernel_size[-1])

  with tf.variable_scope('octree_conv'):
    dim = int(data.shape[1]) * kernel_size[0] * kernel_size[1] * kernel_size[2]
    kernel = tf.get_variable('weights', shape=[channel, dim], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    conv = _octree_conv(data, kernel, octree, depth, channel, kernel_size, stride)
    if stride == 2:
      conv = octree_pad(conv, octree, depth-1)
  return conv


def octree_deconv_fast(data, octree, depth, channel, kernel_size=[3], stride=1):
  assert(type(kernel_size) is list and len(kernel_size) < 4)
  for i in range(len(kernel_size), 3):
    kernel_size.append(kernel_size[-1])

  with tf.variable_scope('octree_deconv'):
    kernel_sdim = kernel_size[0] * kernel_size[1] * kernel_size[2]
    dim = channel * kernel_sdim
    kernel = tf.get_variable('weights', shape=[int(data.shape[1]), dim], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    if stride == 2:
      data = octree_depad(data, octree, depth)
      depth = depth + 1
    data = tf.squeeze(data, [0, 3])
    deconv = tf.matmul(kernel, data, transpose_a=True, transpose_b=False)
    deconv = tf.reshape(deconv, [channel, kernel_sdim, -1])
    col = col2octree(deconv, octree, depth, kernel_size, stride)
  return col


def octree_deconv_memory(data, octree, depth, channel, kernel_size=[3], stride=1):
  assert(type(kernel_size) is list and len(kernel_size) < 4)
  for i in range(len(kernel_size), 3):
    kernel_size.append(kernel_size[-1])

  with tf.variable_scope('octree_deconv'):
    kernel_sdim = kernel_size[0] * kernel_size[1] * kernel_size[2]
    dim = channel * kernel_sdim
    kernel = tf.get_variable('weights', shape=[int(data.shape[1]), dim], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    if stride == 2:
      data = octree_depad(data, octree, depth)
    deconv = _octree_deconv(data, kernel, octree, depth, channel, kernel_size, stride)
  return deconv


def octree_full_voxel(data, depth):
  height = 2 ** (3 * depth)
  channel = int(data.shape[1])
  with tf.variable_scope('octree_full_voxel'):    
    data = tf.reshape(data, [channel, -1, height]) # (1, C, H, 1) -> (C, batch_size, H1)
    data = tf.transpose(data, perm=[1, 0, 2])
  return data


def octree_tile(data, octree, depth):
  with tf.variable_scope('octree_tile'):
    data = octree_depad(data, octree, depth) # (1, C, H, 1)
    data = tf.tile(data, [1, 1, 1, 8])       # (1, C, H, 8)
    channel = int(data.shape[1])
    output = tf.reshape(data, [1, channel, -1, 1])
  return output


def octree_global_pool(data, octree, depth):
  with tf.variable_scope('octree_global_pool'):
    segment_ids = octree_property(octree, property_name='index', dtype=tf.int32,
                                  depth=depth, channel=1)
    segment_ids = tf.reshape(segment_ids, [-1])
    data = tf.squeeze(data, axis=[0, 3])             # (1, C, H, 1) -> (C, H)
    data = tf.transpose(data)                        # (C, H) -> (H, C)
    output = tf.math.segment_mean(data, segment_ids) # (H, C) -> (batch_size, C)
  return output


def octree_bilinear_legacy(data, octree, depth, target_depth):
  with tf.variable_scope('octree_bilinear'):
    mask = tf.constant(
      [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], 
       [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.float32)
    index, fracs = _octree_bilinear(octree, depth, target_depth)
    feat = tf.transpose(tf.squeeze(data, [0, 3]))        # (1, C, H, 1) -> (H, C)
    output = tf.zeros([tf.shape(index)[0], tf.shape(feat)[1]], dtype=tf.float32)
    norm   = tf.zeros([tf.shape(index)[0], 1], dtype=tf.float32)
    for i in range(8):
      idxi = index[:, i]
      weight = tf.abs(tf.reduce_prod(mask[i, :] - fracs, axis=1, keepdims=True))
      output += weight * tf.gather(feat, idxi) 
      norm   += weight * tf.expand_dims(tf.cast(idxi > -1, dtype=tf.float32), -1)
    output = tf.div(output, norm)
    output = tf.expand_dims(tf.expand_dims(tf.transpose(output), 0), -1)
  return output


# pts: (N, 4), i.e. N x (x, y, z, id)
# data: (1, C, H, 1)
def octree_bilinear_v1(pts, data, octree, depth):
  with tf.variable_scope('octree_bilinear'):
    mask = tf.constant(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.float32)

    xyzf, ids = tf.split(pts, [3, 1], 1)
    xyzf = xyzf - 0.5     # since the value is defined on the center of each voxel
    xyzi = tf.floor(xyzf) # the integer part
    frac = xyzf - xyzi    # the fraction part

    feat = tf.transpose(tf.squeeze(data, [0, 3]))        # (1, C, H, 1) -> (H, C)
    output = tf.zeros([tf.shape(xyzi)[0], tf.shape(feat)[1]], dtype=tf.float32)
    norm   = tf.zeros([tf.shape(xyzi)[0], 1], dtype=tf.float32)

    for i in range(8):
      maski = mask[i, :]
      maskc = 1.0 - maski
      xyzm = xyzi + maski
      xyzm = tf.cast(tf.concat([xyzm, ids], axis=1), dtype=tf_uints)
      idxi = octree_search_key(octree_encode_key(xyzm), octree, depth, is_xyz=True)

      weight = tf.abs(tf.reduce_prod(maskc - frac, axis=1, keepdims=True))
      output += weight * tf.gather(feat, idxi)
      norm += weight * tf.expand_dims(tf.cast(idxi > -1, dtype=tf.float32), -1)
    output = tf.div(output, norm)

    output = tf.expand_dims(tf.expand_dims(tf.transpose(output), 0), -1)
    frac = tf.expand_dims(tf.expand_dims(tf.transpose(frac), 0), -1)

  return output, frac

# pts: (N, 4), i.e. N x (x, y, z, id)
# data: (1, C, H, 1)
def octree_bilinear_v2(pts, data, octree, depth):
  with tf.variable_scope('octree_bilinear'):
    mask = tf.constant(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.float32)

    xyzf, ids = tf.split(pts, [3, 1], 1)
    xyzf = xyzf - 0.5     # since the value is defined on the center of each voxel
    xyzi = tf.floor(xyzf) # the integer part
    frac = xyzf - xyzi    # the fraction part

    output = tf.zeros([1, tf.shape(data)[1], tf.shape(xyzi)[0], 1], dtype=tf.float32)
    norm   = tf.zeros([tf.shape(xyzi)[0], 1], dtype=tf.float32)

    for i in range(8):
      maski = mask[i, :]
      maskc = 1.0 - maski
      xyzm = xyzi + maski
      xyzm = tf.cast(tf.concat([xyzm, ids], axis=1), dtype=tf_uints)
      # !!! Note some elements of idxi may be -1
      idxi = octree_search_key(octree_encode_key(xyzm), octree, depth, is_xyz=True)

      weight = tf.abs(tf.reduce_prod(maskc - frac, axis=1, keepdims=True))
      # output += weight * tf.gather(data, idxi, axis=2)
      output += weight * octree_gather(data, idxi)
      norm   += weight * tf.expand_dims(tf.cast(idxi > -1, dtype=tf.float32), -1)
    output = tf.div(output, norm)
  return output


# pts: (N, 4), i.e. N x (x, y, z, id).
# data: (1, C, H, 1)
# !!! Note: the pts should be scaled into [0, 2^depth]
def octree_bilinear_v3(pts, data, octree, depth):
  with tf.variable_scope('octree_linear'):
    mask = tf.constant(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.float32)
    if octree_key64:
      masku = tf.constant([0, 4294967296, 65536, 4295032832,
                           1, 4294967297, 65537, 4295032833], dtype=tf.int64)
    else:
      masku = tf.constant([0, 65536, 256, 65792,
                           1, 65537, 257, 65793], dtype=tf.int32)

    maskc = 1 - mask

    xyzf, ids = tf.split(pts, [3, 1], 1)
    xyzf = xyzf - 0.5     # since the value is defined on the center of each voxel
    xyzi = tf.floor(xyzf) # the integer part  (N, 3)
    frac = xyzf - xyzi    # the fraction part (N, 3)

    key = tf.cast(tf.concat([xyzi, ids], axis=1), dtype=tf_uints)
    key = tf.cast(octree_encode_key(key), dtype=tf_intk)
    # Cast the key to `int32` since the `add` below does not support `uint64`
    # The size effect is that the batch_size must be smaller than 128
    key = tf.expand_dims(key, 1) + masku  # (N, 8),
    key = tf.cast(tf.reshape(key, [-1]), dtype=tf_uintk)

    idx = octree_search_key(key, octree, depth)  # (N*8,)
    flgs = idx > -1  # filtering flags
    idx = tf.boolean_mask(idx, flgs)

    npt = tf.shape(xyzi)[0]
    ids = tf.reshape(tf.range(npt), [-1, 1])
    ids = tf.reshape(tf.tile(ids, [1, 8]), [-1])  # (N*8,)
    ids = tf.boolean_mask(ids, flgs)

    frac = maskc - tf.expand_dims(frac, axis=1)
    weight = tf.abs(tf.reshape(tf.reduce_prod(frac, axis=2), [-1]))
    weight = tf.boolean_mask(weight, flgs)

    indices = tf.concat([tf.expand_dims(ids, 1), tf.expand_dims(idx, 1)], 1)
    indices = tf.cast(indices, tf.int64)
    data = tf.squeeze(data, [0, 3])  # (C, H)
    h = tf.shape(data)[1]
    mat = tf.SparseTensor(indices=indices, values=weight, dense_shape=[npt, h])

    # channel, max_channel = int(data.shape[0]), 512
    # if channel > max_channel:
    #   num = channel // max_channel
    #   remain = channel % max_channel
    #   splits = [max_channel] * num
    #   if remain != 0:
    #     splits.append(remain)
    #     num += 1
    #   output_split = [None] * num
    #   data_split = tf.split(data, splits, axis=0)
    #   for i in range(num):
    #     with tf.name_scope('mat_%d' % i):
    #       output_split[i] = tf.sparse.sparse_dense_matmul(
    #           mat, data_split[i], adjoint_a=False, adjoint_b=True)
    #   output = tf.concat(output_split, axis=1)
    # else:
    #   output = tf.sparse.sparse_dense_matmul(mat, data, adjoint_a=False, adjoint_b=True)

    output = tf.sparse.sparse_dense_matmul(mat, data, adjoint_a=False, adjoint_b=True)
    norm = tf.sparse.sparse_dense_matmul(mat, tf.ones([h, 1]))
    output = tf.div(output, norm + 1.0e-10) # avoid dividing by zeros
    output = tf.expand_dims(tf.expand_dims(tf.transpose(output), 0), -1)
  return output


def octree_bilinear(data, octree, depth, target_depth, mask=None):
  with tf.name_scope('Octree_bilinear'):
    xyz = octree_property(octree, property_name='xyz', depth=target_depth,
                          channel=1, dtype=tf_uintk)
    xyz = tf.reshape(xyz, [-1])
    if mask is not None:
      xyz = tf.boolean_mask(xyz, mask)
    xyz = tf.cast(octree_decode_key(xyz), dtype=tf.float32)

    # Attention: displacement 0.5, scale
    scale = 2.0**(depth-target_depth)
    xyz += tf.constant([0.5, 0.5, 0.5, 0.0], dtype=tf.float32)
    xyz *= tf.constant([scale, scale, scale, 1.0], dtype=tf.float32)

    output = octree_bilinear_v3(xyz, data, octree, depth)
  return output


# pts: (N, 4), i.e. N x (x, y, z, id)
# data: (1, C, H, 1)
def octree_nearest_interp(pts, data, octree, depth):
  with tf.variable_scope('octree_nearest_interp'):
    # The value is defined on the center of each voxel,
    # so we can get the closest grid point by simply casting the value to tf_uints
    pts = tf.cast(pts, dtype=tf_uints)
    key = tf.reshape(octree_encode_key(pts), [-1])

    idx = octree_search_key(key, octree, depth)
    # !!! Note that some of idx may be -1 or over-bound
    # Use tf.gather may be problematic with some version of tensorflow
    # according to my experiments. So I implemented octree_gather to
    # replace the original tf.gather. If you encounter errors, please
    # use the octree_gather
    # output = tf.gather(data, idx, axis=2)
    output = octree_gather(data, idx)
  return output



def octree_signal(octree, depth, channel):
  with tf.name_scope('octree_signal'):
    signal = octree_property(octree, property_name='feature', dtype=tf.float32,
                             depth=depth, channel=channel)
    signal = tf.reshape(signal, [1, channel, -1, 1])
  return signal


def octree_xyz(octree, depth, decode=True):
  with tf.name_scope('octree_xyz'):
    xyz = octree_property(octree, property_name='xyz', dtype=tf_uintk,
                          depth=depth, channel=1)
    xyz = tf.reshape(xyz, [-1])  # uint32, N
    if decode:
      xyz = octree_decode_key(xyz)  # uint8, Nx4
  return xyz


def octree_child(octree, depth):
  with tf.name_scope('octree_child'):
    child = octree_property(octree, property_name='child', dtype=tf.int32,
                            depth=depth, channel=1)
    child = tf.reshape(child, [-1])
  return child


def octree_split(octree, depth):
  with tf.name_scope('octree_split'):
    split = octree_property(octree, property_name='split', dtype=tf.float32,
                            depth=depth, channel=1)
    split = tf.reshape(split, [-1])
  return split