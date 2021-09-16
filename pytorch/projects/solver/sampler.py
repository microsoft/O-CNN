import torch
from torch.utils.data import Sampler, DistributedSampler, Dataset


class InfSampler(Sampler):
  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    self.dataset = dataset
    self.shuffle = shuffle
    self.reset_sampler()

  def reset_sampler(self):
    num = len(self.dataset)
    indices = torch.randperm(num) if self.shuffle else torch.arange(num)
    self.indices = indices.tolist()
    self.iter_num = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value

  def __len__(self):
    return len(self.dataset)


class DistributedInfSampler(DistributedSampler):
  def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
    super().__init__(dataset, shuffle=shuffle)
    self.reset_sampler()

  def reset_sampler(self):
    self.indices = list(super().__iter__())
    self.iter_num = 0

  def __iter__(self):
    return self

  def __next__(self):
    value = self.indices[self.iter_num]
    self.iter_num = self.iter_num + 1

    if self.iter_num >= len(self.indices):
      self.reset_sampler()
    return value
