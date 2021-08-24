import numpy as np
import torch
import ocnn


class Points2Octree:
  ''' Convert a point cloud into an octree
  '''

  def __init__(self, depth, full_depth=2, node_dis=False, node_feature=False,
               split_label=False, adaptive=False, adp_depth=4, th_normal=0.1,
               th_distance=2.0, extrapolate=False, save_pts=False, key2xyz=False,
               **kwargs):
    self.depth = depth
    self.full_depth = full_depth
    self.node_dis = node_dis
    self.node_feature = node_feature
    self.split_label = split_label
    self.adaptive = adaptive
    self.adp_depth = adp_depth
    self.th_normal = th_normal
    self.th_distance = th_distance
    self.extrapolate = extrapolate
    self.save_pts = save_pts
    self.key2xyz = key2xyz

  def __call__(self, points):
    octree = ocnn.points2octree(points, self.depth, self.full_depth, self.node_dis,
                                self.node_feature, self.split_label, self.adaptive,
                                self.adp_depth, self.th_normal, self.th_distance,
                                self.extrapolate, self.save_pts, self.key2xyz)
    return octree


class NormalizePoints:
  ''' Normalize a point cloud with its bounding sphere

  Args: 
      bsphere: The method used to calculate the bounding sphere, choose from
               'sphere' (bounding sphere) or 'box' (bounding box).
      radius:  Mannually specify the radius of the bounding sphere, -1 means 
               that the bounding sphere is not provided.
  '''

  def __init__(self, bsphere='sphere', radius=-1.0, center=(-1.0,), **kwargs):
    self.bsphere = bsphere
    self.radius = radius
    self.center = center

  def __call__(self, points):
    if self.radius < 0:
      bsphere = ocnn.bounding_sphere(points, self.bsphere)
      radius, center = bsphere[0], bsphere[1:]
    else:
      radius, center = self.radius, self.center
    points = ocnn.normalize_points(points, radius, center)
    return points


class TransformPoints:
  ''' Transform a point cloud and the points out of [-1, 1] are dropped. Make
  sure that the input points are in [-1, 1]

  '''

  def __init__(self, distort, angle=[0, 180, 0], scale=0.25, jitter=0.25,
               offset=0.0, angle_interval=[1, 1, 1], uniform_scale=False,
               normal_axis='', **kwargs):
    self.distort = distort
    self.angle = angle
    self.scale = scale
    self.jitter = jitter
    self.offset = offset
    self.angle_interval = angle_interval
    self.uniform_scale = uniform_scale
    self.normal_axis = normal_axis

  def __call__(self, points):
    rnd_angle = [0.0, 0.0, 0.0]
    rnd_scale = [1.0, 1.0, 1.0]
    rnd_jitter = [0.0, 0.0, 0.0]
    if self.distort:
      mul = 3.14159265 / 180.0
      for i in range(3):
        rot_num = self.angle[i] // self.angle_interval[i]
        rnd = np.random.randint(low=-rot_num, high=rot_num+1, dtype=np.int32)
        rnd_angle[i] = rnd * self.angle_interval[i] * mul

      minval, maxval = 1 - self.scale, 1 + self.scale
      rnd_scale = np.random.uniform(low=minval, high=maxval, size=(3)).tolist()
      if self.uniform_scale:
        rnd_scale = [rnd_scale[0]]*3

      minval, maxval = -self.jitter, self.jitter
      rnd_jitter = np.random.uniform(low=minval, high=maxval, size=(3)).tolist()

    # The range of points is [-1, 1]
    points = ocnn.transform_points(
        points, rnd_angle, rnd_scale, rnd_jitter, self.offset, self.normal_axis)
    # clip the points into [-1, 1]
    points, inbox_mask = ocnn.clip_points(points, [-1.0]*3, [1.0]*3)
    return points, inbox_mask


class TransformCompose:
  def __init__(self, flags):
    self.flags = flags
    self.normalize_points =  NormalizePoints(**flags)
    self.transform_points = TransformPoints(**flags)
    self.points2octree = Points2Octree(**flags)

  def __call__(self, points, idx):
    # Normalize the points into one unit sphere in [-1, 1]
    points = self.normalize_points(points)

    # Apply the general transformations provided by ocnn.
    # The augmentations including rotation, scaling, and jittering, and the
    # input points out of [-1, 1] are clipped
    points, inbox_mask = self.transform_points(points)

    # Convert the points in [-1, 1] to an octree
    octree = self.points2octree(points)

    return {'octree': octree, 'points': points, 'inbox_mask': inbox_mask}


def collate_octrees(batch):
  assert type(batch) == list

  outputs = {}
  for key in batch[0].keys():
    outputs[key] = [b[key] for b in batch]

    # Merge a batch of octrees into one super octree
    if 'octree' in key:
      outputs[key] = ocnn.octree_batch(outputs[key])

    # Convert the labels to a Tensor
    if 'label' in key:
      outputs['label'] = torch.tensor(outputs[key])

    # # Concat the inbox_mask
    # if 'inbox_mask' in key:
    #   pt_num = [mk.numel() for mk in outputs['inbox_mask']]
    #   outputs['pt_num'] = torch.tensor(pt_num)
    #   outputs['inbox_mask'] = torch.cat(outputs['inbox_mask'], dim=0)

  return outputs
