import os
import torch
import numpy as np


class ModelNet40(torch.utils.data.Dataset):
  def __init__(self, root, train=True, transform=None, in_memory=True):
    super(ModelNet40, self).__init__()
    self.root = root
    self.train = train
    self.transform = transform
    self.in_memory = in_memory
    self.points, self.labels, self.category = self.load_modelnet40()

  def __len__(self):
    return len(self.points)

  def __getitem__(self, idx):
    if self.in_memory:
      points = self.points[idx]
    else:
      points = np.fromfile(self.points[idx], np.uint8)
    points = torch.from_numpy(points) # convert it to torch.tensor
    if self.transform:
      octree = self.transform(points)
      
    return octree, self.labels[idx]

  def load_modelnet40(self, suffix='points'):
    points, labels = [], []
    folders = sorted(os.listdir(self.root))
    assert len(folders) == 40
    for idx, folder in enumerate(folders):
      subfolder = 'train' if self.train else 'test'
      current_folder = os.path.join(self.root, folder, subfolder)
      filenames = sorted(os.listdir(current_folder))
      for filename in filenames:
        if filename.endswith(suffix):
          filename_abs = os.path.join(current_folder, filename)
          if self.in_memory:
            points.append(np.fromfile(filename_abs, dtype=np.uint8))
          else:
            points.append(filename_abs)
          labels.append(idx)
    return points, labels, folders
