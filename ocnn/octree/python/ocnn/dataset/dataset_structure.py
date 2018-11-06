"""Classes for defining dataset structures"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import csv
import os
import six

from ocnn.dataset.file_utils import find_files

@six.add_metaclass(abc.ABCMeta)
class DatasetStructure:
    """Base class defining DatasetStructure."""

    EXPECTED_SPLITS = ('train', 'test', 'val')

    def __init__(self, base_folder, patterns):
        """ Initializes DatasetStructure
        Args:
          base_folder: Base folder of dataset.
          patterns: List of glob patterns of files in dataset.
        """

        self._class_map = None
        self.base_folder = base_folder
        self.patterns = patterns

    @abc.abstractmethod
    def generate_class_map(self):
        """ Generator of dataset items in dataset.
        Yields split, class, and file_path of a dataset item.
        """
        raise NotImplementedError("generate_class_map is not implemented")

class FolderMappedStructure(DatasetStructure):
    """ DatasetStructure of datasets that are organized by folders.
    The folder organization is <split>/<class>/<dataset_item>.
    For example, ModelNet follows this structure.
    """

    def generate_class_map(self):
        """ Generator of dataset items in dataset.
        Yields split, class, and file_path of a dataset item.
        """
        try:
            _, potential_classes, _ = next(os.walk(self.base_folder))
            for potential_class in potential_classes:
                class_folder = os.path.join(self.base_folder, potential_class)
                _, splits, _ = next(os.walk(class_folder))
                for split in splits:
                    if split in DatasetStructure.EXPECTED_SPLITS:
                        split_folder = os.path.join(class_folder, split)
                        for pattern in self.patterns:
                            for file_path in find_files(split_folder, pattern, False):
                                yield split, potential_class, file_path
        except StopIteration:
            raise RuntimeError("Unable to list contents")

class CsvMappedStructure(DatasetStructure):
    """ DatasetStructure of datasets that are organized by CSV.
    The folder organization is <split>/<class>/<dataset_item>.
    For example, ShapeNet follows this structure.
    """

    EXPECTED_HEADER = ('id', 'synsetId', 'subSynsetId', 'modelId', 'split')

    def __init__(self, base_folder, patterns, class_map_path):
        """ Initializes DatasetStructure
        Args:
          base_folder: Base folder of dataset.
          patterns: List of glob patterns of files in dataset.
          class_map_path: Path of CSV class map.
        """
        self.class_map_path = class_map_path
        super(CsvMappedStructure, self).__init__(base_folder, patterns)

    def _get_reader(self, csv_file):
        """ Gets csv reader while checking column headers are as expected.

        Args:
          csv_file: csv file object

        Returns:
          csv.reader object positoned at row after header.
        """
        try:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            if len(header) != len(CsvMappedStructure.EXPECTED_HEADER):
                raise RuntimeError("Num columns of csv is {0}, expected, {1}".format(
                    len(header),
                    len(CsvMappedStructure.EXPECTED_HEADER)))
            for column, expected_column in zip(header, CsvMappedStructure.EXPECTED_HEADER):
                if column != expected_column:
                    raise RuntimeError(
                        "Column name {0} does not match expected column name {1}".format(
                            column,
                            expected_column))

        except StopIteration:
            raise RuntimeError("CSV File {0} is empty".format(self.class_map_path))

    def generate_class_map(self):
        """ Generator of dataset items in dataset.
        Yields split, class, and file_path of a dataset item.
        """
        with open(self.class_map_path, 'r') as csv_file:
            csv_reader = self._get_reader(csv_file)
            for _, synset_id, _, model_id, split in csv_reader:
                synset_id = '0' + synset_id
                if split not in DatasetStructure.EXPECTED_SPLITS:
                    raise RuntimeError("Unknown dataset split {0}".format(split))
                walk_folder = os.path.join(self.base_folder, synset_id, model_id)
                for pattern in self.patterns:
                    for file_path in find_files(walk_folder, pattern, False):
                        yield split, synset_id, file_path
