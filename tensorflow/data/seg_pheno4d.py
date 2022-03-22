import os
from random import random
import shutil
import json
from tracemalloc import start
from sklearn.model_selection import train_test_split

current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(current_path, 'script/dataset/shapenet_segmentation')
ply2points = 'ply2points'
convert_tfrecords = os.path.join(current_path, 'util/convert_tfrecords.py')

txt_folder = os.path.join(root_folder, 'txt_reduced_100000_partial')
ply_folder = os.path.join(root_folder, 'ply_reduced_100000')
points_folder = os.path.join(root_folder, 'points_reduced_100000_partial')
dataset_folder = os.path.join(root_folder, 'datasets_reduced_100000_partial')

categories= ['Tomato', 'Maize']
seg_num   = [2, 2]


def ply_to_points():
  print('Convert ply files to points files ...')
  for c in categories:
    src_folder = os.path.join(ply_folder, c)
    des_folder = os.path.join(points_folder, c)
    list_folder = os.path.join(ply_folder, 'filelist')
    if not os.path.exists(des_folder): os.makedirs(des_folder)
    if not os.path.exists(list_folder): os.makedirs(list_folder)

    list_filename = os.path.join(list_folder, c + '.txt')
    filenames = [os.path.join(src_folder, filename) for filename in os.listdir(src_folder)]
    with open(list_filename, 'w') as fid:
      fid.write('\n'.join(filenames))

    cmds = [ply2points,
          '--filenames', list_filename,
          '--output_path', des_folder,
          '--verbose', '0']
    cmd = ' '.join(cmds)
    print(cmd + '\n')
    os.system(cmd)

def split_data():
  file_list = getListOfFiles(points_folder)
  train, test = train_test_split(file_list, test_size=0.4, random_state = 42)
  return train,test

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(getSafeFileNameWithoutExt(fullPath))
                
    return allFiles

def getSafeFileNameWithoutExt(file:str):
    start_index=-1
    try:
      start_index = file.rindex('Tomato')
    except:
      start_index = file.rindex('Maize')

    end_index = file.rindex('.')
    return file[start_index:end_index]

def points_to_tfrecords():
  print('Convert points files to tfrecords files ...')
  if not os.path.exists(dataset_folder): os.makedirs(dataset_folder)
  list_folder     = os.path.join(txt_folder, 'train_test_split')
  if not os.path.exists(list_folder): os.makedirs(list_folder)
  train, test = split_data()
  for i, c in enumerate(categories):
    filelist_name = os.path.join(list_folder, c + '_train_val.txt')
    filelist = ['%s.points %d' % (line, i) for line in train if c in line] 
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(filelist))

    dataset_name =  os.path.join(dataset_folder, c + '_train_val.tfrecords')
    cmds = ['python', convert_tfrecords,
            '--file_dir', points_folder,
            '--list_file', filelist_name,
            '--records_name', dataset_name]
    cmd = ' '.join(cmds)
    print(cmd + '\n')
    os.system(cmd)

    filelist_name = os.path.join(list_folder, c + '_test.txt')
    filelist = ['%s.points %d' % (line, i) for line in test if c in line]
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(filelist))

    dataset_name = os.path.join(dataset_folder, c + '_test.tfrecords')
    cmds = ['python', convert_tfrecords,
            '--file_dir', points_folder,
            '--list_file', filelist_name,
            '--records_name', dataset_name]
    cmd = ' '.join(cmds)
    print(cmd + '\n')
    os.system(cmd)
  des_folder = os.path.join(root_folder, 'train_test_split')
  if not os.path.exists(des_folder): 
    shutil.copytree(list_folder, des_folder)

if __name__ == '__main__':
  ply_to_points()
  points_to_tfrecords()
  #split_data()