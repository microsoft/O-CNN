"""Module containing abstract interfaces for augmentors """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

class AugmentorCollection:
    """Base class defining Augmentor Collection"""
    def __init__(self, num_aug):
        """ Initializes Augmentor Collection
        Args:
          num_aug: Total number of augmentations to perform on an object.
        """
        self._augmentors = []
        self._num_aug = num_aug

    @property
    def num_aug(self):
        """ Total number of augmentations to perform on an object """
        return self._num_aug

    def augment(self, item, aug_index):
        """ Augment an object
        Args:
          item: Item to augment
          aug_index: Augmentation index to apply modification
        """

        if aug_index >= self.num_aug:
            raise RuntimeError(
                'Invalid aug_index {0}, total aug {1}'.format(aug_index,
                                                              self.num_aug))
        for augmentor in self._augmentors:
            augmentor.augment(item, aug_index)


@six.add_metaclass(abc.ABCMeta)
class Augmentor:
    """Base class defining Augmentor"""
    @abc.abstractmethod
    def augment(self, item, aug_index):
        """Modifies points object
        Args:
          item: item to modify
          aug_index: Augmentation index to apply modification
        """
        raise NotImplementedError("augment is not implemented")
