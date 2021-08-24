import ocnn
import torch
import pickle
import numpy as np

from solver import Dataset


class ScanOctree:
  def __init__(self, camera_path, scan=True):
    self.scan = scan
    self.camera_path = camera_path
    if self.scan:
      with open(camera_path, 'rb') as fid:
        self.camera = pickle.load(fid)

  def generate_scan_axis(self, i):
    j = np.random.randint(0, 8)
    key = '%d_%d' % (i, j)
    axes = np.array(self.camera[key])

    # perturb the axes
    rnd = np.random.random(axes.shape) * 0.4 - 0.2
    axes = np.reshape(axes + rnd, (-1, 3))

    # normalize the axes
    axes = axes / np.sqrt(np.sum(axes ** 2, axis=1, keepdims=True) + 1.0e-6)
    axes = np.reshape(axes, (-1)).astype(np.float32).tolist()
    return axes

  def __call__(self, octree, idx):
    if self.scan:
      scan_axis = self.generate_scan_axis(idx)
      partial_octree = ocnn.octree_scan(octree, scan_axis)
      return partial_octree
    else:
      return octree


class ScanTransform(ocnn.TransformCompose):
  def __init__(self, flags):
    super().__init__(flags)
    self.scan_octree = ScanOctree(flags.camera_path, flags.scan)

  def __call__(self, points, idx):
    # apply the default transformation provided by ocnn
    output = super().__call__(points, idx)
    # generate the partial octree via virtual scanning
    output['octree_in'] = self.scan_octree(output['octree'], idx)
    return output


class Noise2cleanTransform:
  # Follow the [preprocess steps](https://github.com/autonomousvision/occupancy_networks#Building-the-dataset)
  # of `Occupancy Networks` to generate the training data.
  def __init__(self, flags):
    self.points_number = 3000
    self.points_scale = 0.95
    self.noise_std = 0.01 * self.points_scale

    self.points2octree = ocnn.Points2Octree(**flags)

  def __call__(self, point_cloud, idx):
    # get the input
    points, normals = point_cloud['points'], point_cloud['normals']

    # normalize the points
    bbmin, bbmax = np.min(points, axis=0), np.max(points, axis=0)
    center = (bbmin + bbmax) / 2.0
    radius = 2.0 / (np.max(bbmax - bbmin) + 1.0e-6)
    points = (points - center) * radius  # normalize to [-1, 1]
    points *= self.points_scale  # normalize to [-points_scale, points_scale]

    # randomly sample points and add noise
    noise = self.noise_std * np.random.randn(self.points_number, 3)
    rand_idx = np.random.choice(points.shape[0], size=self.points_number)
    points_noise = points[rand_idx] + noise

    # transform points to octree
    points_gt = ocnn.points_new(
        torch.from_numpy(points).float(), torch.from_numpy(normals).float(),
        torch.Tensor(),  torch.Tensor())
    points_gt, _ = ocnn.clip_points(points_gt, [-1.0]*3, [1.0]*3)
    octree_gt = self.points2octree(points_gt)

    points_in = ocnn.points_new(
        torch.from_numpy(points_noise).float(), torch.Tensor(),
        torch.ones(self.points_number).float(), torch.Tensor())
    points_in, _ = ocnn.clip_points(points_in, [-1.0]*3, [1.0]*3)
    octree_in = self.points2octree(points_in)

    return {'octree_in': octree_in, 'points_in': points_in,
            'octree': octree_gt,    'points': points_gt}


def get_completion_dataset(flags):
  if flags.name == 'completion':
    transform = ScanTransform(flags)
  elif flags.name == 'noise2clean':
    transform = Noise2cleanTransform(flags)
  else:
    raise ValueError

  dataset = Dataset(flags.location, flags.filelist, transform,
                    in_memory=flags.in_memory)
  return dataset, ocnn.collate_octrees
