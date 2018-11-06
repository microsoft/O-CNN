"""Class which creates WritableData objects""" 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

@six.add_metaclass(abc.ABCMeta)
class DataProcessor:
    """Base class defining DataProcessor."""

    def __init__(self, augmentors=None):
        """ Initializes DataProcessor
        Args:
          augmentors: List of augmentors before data is processed
        """
        self.augmentors = augmentors if augmentors else []

    @abc.abstractmethod
    def process(self, file_path, aug_index):
        """ Augments input data and returns WritableData object
        Args:
          file_path: Path of input object
          aug_index: Augmentation index of total augmentations.
        Returns:
          WritableData object
        """

        raise NotImplementedError("process is not implemented")
