from abc import ABCMeta, abstractproperty
from dataset import Dataset
from enum import Enum
from file_utils import find_files, write_out_iterable

import off_tools
import os
import shutil

class ModelNet(Dataset):
    """
        Implementation of Dataset for ModelNet datasets.
    """

    __metaclass__ = ABCMeta

    def class_file_map_generator(self, file_list, class_map):
        """
            Generates classification map string used to create LMDB.
            The generated string should be of the form:
                <file_name.octree> <category_number>

            Args:
                file_list (iterable of str): File list of octrees to generate classification maps.
                class_map (iterable of str): Class map which maps file path substring to class id

            Yields:
                str: Generated string of the form <file_name.octree> <category_number>

        """
        for file_path in file_list:
            class_map_iter = iter(class_map)
            while True:
                try:
                    search_path = next(class_map_iter)
                    if search_path in file_path:
                        break
                except StopIteration:
                    raise RuntimeError("File {0} does not correspond to dataset's classes")
            rel_path = os.path.relpath(file_path, self.dataset_directory)
            yield rel_path + ' ' + class_map[search_path] + '\n'

    def clean_dataset(self):
        """
            Cleans malformed header in OFF files
        """

        super(ModelNet, self).clean_dataset()
        off_tools.clean_dataset(self.dataset_directory)
        print("Dataset cleaned")

    def prepare_octree_file_list(self):
        """
            Creates file list in dataset directory. File list is composed of all octree files and
            their corresponding classification category.
            The file list should be of the form:
                <file_name.octree> <category_number>
        """

        model_directory_path = os.path.join(self.dataset_directory,
                                            self.dataset_filename_stem)
        class_list = os.listdir(model_directory_path)
        class_map = self._generate_class_map(class_list)

        test_list = find_files(self.dataset_directory, '*.octree')
        train_list = find_files(self.dataset_directory, '*.octree')

        test_list_generator = self.class_file_map_generator(test_list, class_map)
        train_list_generator = self.class_file_map_generator(train_list, class_map)

        write_out_iterable(self.octree_train_list_path, train_list_generator)
        write_out_iterable(self.octree_test_list_path, test_list_generator)

    def _generate_class_map(self, class_list):
        """
            Generates list of class maps which maps file path substring to class id

            Args:
              class_list (iterable of str): List of classes to categorize

            Returns:
              Dictionary which maps searchable substring to a class id
        """
        class_map = {}
        for idx, class_type in enumerate(class_list):
            search_path = os.path.join(os.sep, class_type, '')
            class_map[search_path] = str(idx)
        return class_map

class ModelNet40(ModelNet):
    """
        ModelNet40 Dataset
    """
    @property
    def archive_uri(self):
        return "http://modelnet.cs.princeton.edu/ModelNet40.zip"

class ModelNet10(ModelNet):
    """
        ModelNet10 Dataset
    """
    @property
    def archive_uri(self):
        return "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"

    def clean_dataset(self):
        """
            Cleans malformed header in OFF files
        """
        print("Removing erroneous files")
        shutil.rmtree(os.path.join(self.dataset_directory, '__MACOSX'))

        super(ModelNet10, self).clean_dataset()
