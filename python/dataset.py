from __future__ import print_function

from abc import ABCMeta, abstractproperty, abstractmethod
from enum import Enum
from file_utils import write_out_iterable, find_files, line_separator_generator
from shutil import copyfile

import os
import pickle
import subprocess
import zipfile

try:
    import urllib.request as urllib
except ImportError:
    import urllib

class DatasetActions(Enum):
    """
        Actions that the dataset object can perform
    """
    Retrieve = 1
    Extract = 2
    Clean = 3
    CreatePoints = 4
    CreateOctree = 5
    CreateLmdb = 6
    Finished = 7
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __str__(self):
        return str(self.name)

class ConversionArguments:
    """
        Arguments required to convert the dataset to octrees
    """
    def __init__(self, depth, full_layer, displacement,
                 augmentation, for_segmentation):
        self.depth = depth
        self.full_layer = full_layer
        self.displacement = displacement
        self.augmentation = augmentation
        self.for_segmentation = for_segmentation
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return (self.depth == other.depth and
                    self.full_layer == other.full_layer and
                    self.displacement == other.displacement and
                    self.augmentation == other.augmentation and
                    self.for_segmentation == other.for_segmentation)
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return ('Depth: ' + str(self.depth) + os.linesep +
                'Full Layer: ' + str(self.full_layer) + os.linesep +
                'Displacement: ' + str(self.displacement) + os.linesep +
                'Augmentation: ' + str(self.full_layer) + os.linesep +
                'For Segmentation: ' + str(self.for_segmentation) + os.linesep)

class Dataset:
    """
        Dataset class to create octree files and LMDB files for training / testing Caffe models.
    """
    __metaclass__ = ABCMeta
    @abstractproperty
    def archive_uri(self):
        """
            Uri of archived dataset
        """
        pass

    @abstractmethod
    def prepare_octree_file_list(self):
        """
            Creates file list in dataset directory. File list is composed of all octree files and
            their corresponding classification category.
            The file list should be of the form:
                <file_name.octree> <category_number>
        """
        pass

    CACHED_ACTIONS_FILE = 'cached_actions.pkl'
    POINTS_LIST_FILE = 'points_list.txt'
    OCTREE_TRAIN_LIST_FILE = 'oct_list_train.txt'
    OCTREE_TEST_LIST_FILE = 'oct_list_test.txt'
    OCTREE_TRAIN_DB_FOLDER = 'train_db'
    OCTREE_TEST_DB_FOLDER = 'test_db'

    @staticmethod
    def _get_conversion_error_str(cached_conversion_arguments, specified_conversion_arguments):
        """
            Generates error string for mismatched conversion arguments

            Args:
                cached_conversion_arguments (str): Conversion arguments which were cached
                specified_conversion_arguments (str): Conversion arguments which were specified
            Returns:
                str: Conversion error string
        """
        return ("Cached conversion arguments:" + os.linesep + os.linesep +
                str(cached_conversion_arguments) + os.linesep +
                "differs from specified arguments:" + os.linesep + os.linesep +
                str(specified_conversion_arguments) + os.linesep)

    def __init__(self, data_directory):
        """
            Initializes Dataset object

            Args:
                data_directory (str): Top level directory to extract dataset into.

            Attributes:
                data_directory (str): Top level directory to extract dataset into.
                archive_uri (str): Uri of archived dataset
                dataset_filename (str): Filename of archived dataset
                dataset_filename_stem (str): Filename of archive dataset without extension
                dataset_archive_path (str): Path to copy archived dataset to.
                dataset_directory (str): Path to extract archived dataset to.
                cached_actions_path (str): Path to pickled file with any previously cached actions.
                octree_train_list_path (str): Path to file list of octree files for training
                octree_test_list_path (str): Path to file list of octree files for testing
        """
        self.data_directory = data_directory
        self.dataset_filename = os.path.basename(self.archive_uri)
        self.dataset_filename_stem, _ = os.path.splitext(self.dataset_filename)
        self.dataset_archive_path = os.path.join(self.data_directory, self.dataset_filename)
        self.dataset_directory = os.path.join(self.data_directory, self.dataset_filename_stem, '')
        if not os.path.exists(self.dataset_directory):
            os.makedirs(self.dataset_directory)

        self.cached_actions_path = os.path.join(self.dataset_directory, Dataset.CACHED_ACTIONS_FILE)
        self.octree_train_list_path = os.path.join(self.dataset_directory,
                                                   Dataset.OCTREE_TRAIN_LIST_FILE)
        self.octree_test_list_path = os.path.join(self.dataset_directory,
                                                  Dataset.OCTREE_TEST_LIST_FILE)

    def get_initial_action(self, conversion_arguments, starting_action):
        """
            Determines initial action to take with dataset when automatically preparing dataset.

            Args:
                conversion_arguments (ConversionArguments): Arguments specified to convert to octree
                starting_action (DatasetActions): starting action to start when preparing dataset.

            Returns:
                DatasetActions: Action to perform when preparing the dataset.

            Raises:
                AttributeError: Any error in the conversion arguments or starting action which
                would break the conversion process.
        """
        try:
            next_action, cached_conversion_arguments = self.get_cached_data()
        except IOError:
            next_action = DatasetActions.Retrieve
            cached_conversion_arguments = conversion_arguments

        if starting_action:
            if starting_action <= next_action:
                next_action = starting_action
            else:
                raise AttributeError("starting action {0} is ahead of cached action {1}")

        if cached_conversion_arguments != conversion_arguments:
            if next_action <= DatasetActions.CreateOctree:
                if (find_files(self.dataset_directory,
                               "*.octree",
                               find_first_instance=True)):
                    raise AttributeError(
                        Dataset._get_conversion_error_str(
                            cached_conversion_arguments,
                            conversion_arguments)
                        + "remove octree file or use same conversion arguments")
            else:
                raise AttributeError(
                    Dataset._get_conversion_error_str(
                        cached_conversion_arguments,
                        conversion_arguments)
                    + "specify earlier action or use same conversion arguments")

        return next_action

    def prepare_set(self,
                    conversion_arguments,
                    point_converter_path,
                    octree_converter_path,
                    lmdb_converter_path,
                    starting_action=None):
        """
            Prepares the dataset by outputting training and testing LMDB files

            Args:
                conversion_arguments (ConversionArguments): Arguments specified to convert to octree
                point_converter_path (str): Path to obj/off to points converter executable
                octree_converter_path (str): Path to points to octree converter executable
                lmdb_converter_path (str): Path to octree to lmdb converter executable.
                starting_action (DatasetActions): Specified starting action to be performed
        """

        next_action = self.get_initial_action(conversion_arguments, starting_action)

        if next_action is DatasetActions.Retrieve:
            print('Retrieving dataset')
            self.retrieve_set()
            next_action = DatasetActions.Extract
            self.cache_data(next_action, conversion_arguments)
            print('Retrieved dataset')
        else:
            print('Dataset previously retrieved')

        if next_action is DatasetActions.Extract:
            print('Extracting dataset')
            self.extract_set()
            next_action = DatasetActions.Clean
            self.cache_data(next_action, conversion_arguments)
            print('Extracted dataset')
        else:
            print('Dataset already extracted')

        if next_action is DatasetActions.Clean:
            print('Cleaning dataset')
            self.clean_dataset()
            next_action = DatasetActions.CreatePoints
            self.cache_data(next_action, conversion_arguments)
            print('Cleaned dataset')
        else:
            print('Dataset already cleaned')

        if next_action is DatasetActions.CreatePoints:
            print('Converting to points')
            self.convert_to_points(point_converter_path)
            next_action = DatasetActions.CreateOctree
            self.cache_data(next_action, conversion_arguments)
            print('Converted to points')
        else:
            print('Points files already created')

        if next_action is DatasetActions.CreateOctree:
            print('Converting to octree')
            self.convert_to_octree(octree_converter_path, conversion_arguments)
            next_action = DatasetActions.CreateLmdb
            self.cache_data(next_action, conversion_arguments)
            print('Converted to octree')
        else:
            print('Octree files already created')

        if next_action is DatasetActions.CreateLmdb:
            print('Converting to lmdb')
            self.convert_to_lmdb(lmdb_converter_path)
            next_action = DatasetActions.Finished
            self.cache_data(next_action, conversion_arguments)
            print('Converted to lmdb')
        else:
            print('lmdb files already created')

    def retrieve_set(self):
        """
            Retrieves archived dataset specified by archive uri and copies to data directory.
        """
        if 'http://' in self.archive_uri or 'https://' in self.archive_uri:
            urllib.urlretrieve(self.archive_uri, filename=self.dataset_archive_path)
        else:
            copyfile(self.archive_uri, self.dataset_archive_path)

    def extract_set(self):
        """
            Extracts archived dataset to data directory
        """
        with zipfile.ZipFile(os.path.join(self.data_directory, self.dataset_filename)) as zip_file:
            zip_file.extractall(self.dataset_directory)

    def clean_dataset(self):
        """
            Performs any necessary cleaning operations for a dataset
        """
        pass

    def convert_to_points(self, point_converter_path):
        """
            Converts dataset to points files

            Args:
                point_converter_path (str): Path to obj/off to points converter.
        """
        view_num = '6'
        flag = False
        normalize = True
        subprocess.check_call([point_converter_path,
                               self.dataset_directory,
                               view_num,
                               str(int(flag)),
                               str(int(normalize))])

    def convert_to_octree(self, octree_converter_path, conversion_arguments):
        """
            Converts points files to octrees

            Args:
                octree_converter_path (str): Path to points to octree converter executable
                conversion_arguments (ConversionArguments): Arguments specified to convert to octree
        """
        #check for filelist
        points_list_path = os.path.join(self.dataset_directory, Dataset.POINTS_LIST_FILE)
        file_list = line_separator_generator(find_files(self.dataset_directory, '*.points'), use_os_sep=False)
        write_out_iterable(points_list_path, file_list)
        subprocess.check_call([octree_converter_path,
                               points_list_path,
                               str(conversion_arguments.depth),
                               str(conversion_arguments.full_layer),
                               str(conversion_arguments.displacement),
                               str(conversion_arguments.augmentation),
                               str(int(conversion_arguments.for_segmentation))])

    def convert_to_lmdb(self, lmdb_converter_path):
        """
            Converts octree files to lmdb files
            Args:
                lmdb_converter_path (str): Path to octree to lmdb converter executable.
        """
        octree_train_db_path = os.path.join(self.dataset_directory,
                                            Dataset.OCTREE_TRAIN_DB_FOLDER)
        octree_test_db_path = os.path.join(self.dataset_directory,
                                           Dataset.OCTREE_TEST_DB_FOLDER)

        self.prepare_octree_file_list()


        if (find_files(octree_train_db_path, "*.mdb", find_first_instance=True) or
                find_files(octree_test_db_path, "*.mdb", find_first_instance=True)):
            raise AttributeError(
                "*.mdb file found in dataset path ({0}) please remove them".format(
                    self.dataset_directory))

        subprocess.check_call([lmdb_converter_path,
                               self.dataset_directory,
                               self.octree_train_list_path,
                               octree_train_db_path])
        subprocess.check_call([lmdb_converter_path,
                               self.dataset_directory,
                               self.octree_test_list_path,
                               octree_test_db_path])

    def get_cached_data(self):
        """
            Retrieves pickled cached data used by dataset
        """
        with open(self.cached_actions_path, 'rb') as pickle_file:
            cached_next_action = pickle.load(pickle_file)
            cached_conversion_arguments = pickle.load(pickle_file)
        return cached_next_action, cached_conversion_arguments

    def cache_data(self, action, conversion_arguments):
        """
            Pickels data to be used by dataset when reran.

            Args:
                action (DatasetActions): Action to cache to be performed when reran
                conversion_arguments (ConversionArguments): Arguments to cache

        """
        with open(self.cached_actions_path, 'wb') as pickle_file:
            pickle.dump(action, pickle_file)
            pickle.dump(conversion_arguments, pickle_file)
