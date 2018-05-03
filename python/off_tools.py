from __future__ import print_function

from multiprocessing import Pool
import file_utils
import os

def clean_off_file(file_name):
    """
        Fixes header of OFF file

        Args:
            file_name (str): Name of file to fix
    """
    file_cleaned = False
    with open(file_name) as f_check:
        file_str = f_check.read()
        if file_str[0:3] != 'OFF':
            raise AttributeError('Unexpected Header for {0}'.format(file_name))
        elif file_str[0:4] != 'OFF\n':
            new_str = file_str[0:3] + '\n' + file_str[3:]
            with open(file_name, 'w') as f_rewrite:
                f_rewrite.write(new_str)
            file_cleaned = True
    return file_cleaned

def clean_dataset(input_folder):
    """
        Fixes headers of all OFF files in a given folder.

        Args:
            input_folder (str): Folder to search for off files
    """
    executor = Pool()
    file_list = file_utils.find_files(input_folder, '*.[Oo][Ff][Ff]')

    files_cleaned_list = executor.map(clean_off_file, file_list)
    num_files_cleaned = 0
    for file_cleaned in files_cleaned_list:
        if file_cleaned:
            num_files_cleaned += 1
    print("{0} out of {1} files cleaned".format(num_files_cleaned, len(files_cleaned_list)))
