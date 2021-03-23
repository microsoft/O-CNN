import os
import torch
import numpy as np
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
  def __init__(self, root, filelist, transform=None, in_memory=True):
    super(Dataset, self).__init__()
    self.root = root
    self.filelist = filelist
    self.transform = transform
    self.in_memory = in_memory
    self.samples, self.labels = self.load_dataset()

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    sample = self.samples[idx] if self.in_memory else   \
             np.fromfile(self.samples[idx], dtype=np.uint8)
    sample = torch.from_numpy(sample)  # convert it to torch.tensor
    if self.transform:                 # transform the sample
      sample = self.transform(sample)
    return sample, self.labels[idx]

  def load_dataset(self):
    samples, labels = [], []
    tqdm.write('Load from ' + self.filelist)
    with open(self.filelist) as fid:
      lines = fid.readlines()
    for line in tqdm(lines, ncols=80):
      filename, label = line.split()
      filename_abs = os.path.join(self.root, filename)
      samples.append(np.fromfile(filename_abs, dtype=np.uint8)  \
                     if self.in_memory else filename_abs)
      labels.append(int(label))
    return samples, labels
