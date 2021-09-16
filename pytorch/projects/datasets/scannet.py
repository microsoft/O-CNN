import ocnn
import torch
import numpy as np
import scipy.interpolate
import scipy.ndimage
import random
from plyfile import PlyData

from solver import Dataset


def read_file(filename):
  def _read_ply(filename):
    plydata = PlyData.read(filename)
    vtx = plydata['vertex']
    xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1).astype(np.float32)
    color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1).astype(np.float32)
    label = np.asarray(vtx['label']).astype(np.float32)
    normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1).astype(np.float32)
    return xyz, color, label, normal

  def _read_points(filename):
    points = torch.from_numpy(np.fromfile(filename, dtype=np.uint8))
    xyz = ocnn.points_property(points, 'xyz').numpy()
    label = ocnn.points_property(points, 'label').squeeze().numpy()
    normal = ocnn.points_property(points, 'normal').numpy()
    color = ocnn.points_property(points, 'feature').numpy() * 255  # !!! RGB
    return xyz, color, label, normal

  if filename.endswith('.ply'):
    return _read_ply(filename)
  elif filename.endswith('.points'):
    return _read_points(filename)
  else:
    raise ValueError


def color_distort(color, trans_range_ratio, jitter_std):
  def _color_autocontrast(color, rand_blend_factor=True, blend_factor=0.5):
    assert color.shape[1] >= 3
    lo = color[:, :3].min(0, keepdims=True)
    hi = color[:, :3].max(0, keepdims=True)
    assert hi.max() > 1

    scale = 255 / (hi - lo)
    contrast_feats = (color[:, :3] - lo) * scale

    blend_factor = random.random() if rand_blend_factor else blend_factor
    color[:, :3] = (1 - blend_factor) * color + blend_factor * contrast_feats
    return color

  def _color_translation(color, trans_range_ratio=0.1):
    assert color.shape[1] >= 3
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * trans_range_ratio
      color[:, :3] = np.clip(tr + color[:, :3], 0, 255)
    return color

  def _color_jiter(color, std=0.01):
    if random.random() < 0.95:
      noise = np.random.randn(color.shape[0], 3)
      noise *= std * 255
      color[:, :3] = np.clip(noise + color[:, :3], 0, 255)
    return color

  color = color * 255.0
  color = _color_autocontrast(color)
  color = _color_translation(color, trans_range_ratio)
  color = _color_jiter(color, jitter_std)
  color = color / 255.0
  return color


def elastic_distort(points, distortion_params):
  def _elastic_distort(coords, granularity, magnitude):
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    convolve = scipy.ndimage.filters.convolve
    for _ in range(2):
      noise = convolve(noise, blurx, mode='constant', cval=0)
      noise = convolve(noise, blury, mode='constant', cval=0)
      noise = convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [np.linspace(d_min, d_max, d)
          for d_min, d_max, d in zip(coords_min - granularity,
                                     coords_min + granularity*(noise_dim - 2),
                                     noise_dim)]

    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords

  assert distortion_params.shape[1] == 2
  if random.random() < 0.95:
    for granularity, magnitude in distortion_params:
      points = _elastic_distort(points, granularity, magnitude)
  return points


class TransformScanNet:
  def __init__(self, flags):
    self.flags = flags
    self.scale_factor = 5.12
    self.color_trans_ratio = 0.10
    self.color_jit_std = 0.05
    self.elastic_params = np.array([[0.2, 0.4], [0.8, 1.6]], np.float32)

  def transform_scannet(self, sample):
    # read ply file
    # xyz, color, label, normal = read_file(filename)
    xyz, color, label, normal = sample

    # normalization
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    xyz = (xyz - center) / self.scale_factor  # xyz in [-1, 1]
    color = color / 255.0

    # data augmentation
    if self.flags.distort:
      color = color_distort(color, self.color_trans_ratio, self.color_jit_std)
      xyz = elastic_distort(xyz, self.elastic_params)

    points = ocnn.points_new(torch.from_numpy(xyz), torch.from_numpy(normal),
                             torch.from_numpy(color), torch.from_numpy(label))
    return points

  def __call__(self, sample, idx=None):
    # transformation specified for scannet
    points = self.transform_scannet(sample)

    # general transformations provided by ocnn
    # The augmentations including rotation, scaling, and jittering, and the
    # input points out of [-1, 1] are clipped
    points, inbox_mask = ocnn.TransformPoints(**self.flags)(points)

    # transform points to octree
    octree = ocnn.Points2Octree(**self.flags)(points)
    return {'octree': octree, 'points': points, 'inbox_mask': inbox_mask}


def get_scannet_dataset(flags):
  transform = TransformScanNet(flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, ocnn.collate_octrees
