import numpy as np
import torch
import ocnn


class Points2Octree:
  ''' Convert a point cloud into an octree
  '''

  def __init__(self, depth, full_depth=2, node_dis=False, node_feature=False,
               split_label=False, adaptive=False, adp_depth=4, th_normal=0.1,
               th_distance=1.0, extrapolate=False, save_pts=False, key2xyz=False,
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
      method: The method used to calculate the bounding sphere, choose from
              'sphere' (bounding sphere) or 'box' (bounding box).
  '''

  def __init__(self, method='sphere'):
    self.method = method

  def __call__(self, points):
    bsphere = ocnn.bounding_sphere(points, self.method)
    radius, center = bsphere[0], bsphere[1:]
    points = ocnn.normalize_points(points, radius, center)
    return points


class TransformPoints:
  ''' Transform a point cloud and the points out of [-1, 1] are dropped. Make
  sure that the input points are in [-1, 1]

  '''

  def __init__(self, distort, angle=[0, 180, 0], scale=0.25, jitter=0.25,
               offset=0.0, angle_interval=[1, 1, 1], uniform_scale=False,
               **kwargs):
    self.distort = distort
    self.angle = angle
    self.scale = scale
    self.jitter = jitter
    self.offset = offset
    self.angle_interval = angle_interval
    self.uniform_scale = uniform_scale

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
      if self.uniform_scale:  rnd_scale = [rnd_scale[0]]*3

      minval, maxval = -self.jitter, self.jitter
      rnd_jitter = np.random.uniform(low=minval, high=maxval, size=(3)).tolist()

    # The range of points is [-1, 1]
    points = ocnn.transform_points(points, rnd_angle, rnd_scale, rnd_jitter, self.offset)
    return points


class TransformCompose:
  def __init__(self, flags, return_pts=False):
    self.flags = flags
    self.return_pts = return_pts
  
  def __call__(self, points):
    points = NormalizePoints('sphere')(points)
    points = TransformPoints(**self.flags)(points)
    octree = Points2Octree(**self.flags)(points)
    return octree if not self.return_pts else (octree, points)
    

class CollateOctrees:
  def __init__(self, return_pts=False):
    self.return_pts = return_pts

  def __call__(self, batch):
    ''' Merge a batch of octrees into one super octree
    '''
    assert type(batch) == list 
    octrees = [b[0] for b in batch]
    octree = ocnn.octree_batch(octrees)
    labels = torch.tensor([b[1] for b in batch])

    outputs = [octree, labels]
    if self.return_pts:
      points = [b[2] for b in batch]
      outputs.append(points)
    return outputs


collate_octrees = CollateOctrees(return_pts=False)