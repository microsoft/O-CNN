import os
import argparse
import numpy as np

try:
  from sklearn.cluster import KMeans
except:
  raise ImportError('''
  Error! Can not import sklearn.
  Please install the package sklearn via the following command:

    pip install sklearn    
  ''')

try:
  import pyoctree
except:
  raise ImportError('''\n
  Error! Can not import pyoctree.
  Please build the octree with python enabled via the following commands:  

    cd octree/build
    cmake .. -DUSE_PYTHON=ON  && cmake --build . --config Release
    export PYTHONPATH=`pwd`/python:$PYTHONPATH
  
  ''')


source_parent_folder = "/home/ervin/Desktop/Thesis/Dataset/Plant3DArchitectures/"
points_parent_folder = "/home/ervin/Desktop/Thesis/Dataset/Plant3DSamplePoints/"

def convert_to_points():
    select = 0 # number of files to be preprocessed 0 means all
    folders = os.listdir(source_parent_folder)
    i = 0
    for folder in folders:
        curr_folder = os.path.join(source_parent_folder, folder)
        target_folder = os.path.join(points_parent_folder,folder)
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)
        file_list = getListOfFiles(curr_folder)
        for file in file_list:  
            cmd = 'custom_data --source %s --output_path %s' % (file, target_folder+'/'+getSafeFileName(file).replace('.obj',''))
            print(cmd)
            os.system(cmd)
            i+=1
            if i == select:
                break
        if i == select:
            break

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
            allFiles.append(fullPath)
                
    return allFiles

def getSafeFileName(file:str):
    index = file.rindex('/')
    return file[index+1:]

def plant3d_simplify_points(resolution=128):

  # rename and backup the original folders
  simplified_points_folder = os.path.join(points_parent_folder, 'simplified_points')
  original_folder = points_parent_folder
  simplify = 'simplify_points'
  # if os.path.exists(points_folder):
  #   os.rename(points_folder, original_folder)

  folders = os.listdir(original_folder)
  for folder in folders:
    # write filelist to disk
    curr_folder = os.path.join(original_folder, folder)
    filenames = os.listdir(curr_folder)
    filelist_name = os.path.join(curr_folder, 'list.txt')
    with open(filelist_name, 'w') as fid:
      for filename in filenames:
        if filename.endswith('.points'):
          fid.write(os.path.join(curr_folder, filename) + '\n')

    # run simplify_points
    output_path = os.path.join(simplified_points_folder, folder)
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    cmd = '%s --filenames %s --output_path %s --dim %d' % \
          (simplify
          , filelist_name, output_path, resolution)
    print(cmd)
    os.system(cmd)
    os.remove(filelist_name)

def point_cloud_clustering(point_cloud, n_clusters=100):
  pt_num = point_cloud.pts_num()
  pts = point_cloud.points()
  normals = point_cloud.normals()

  X = np.array(pts).reshape(pt_num, 3)
  if pt_num < n_clusters:
    n_clusters = pt_num
  y_pred = KMeans(n_clusters=n_clusters, n_init=1).fit_predict(X)

  succ = point_cloud.set_points(pts, normals, [], y_pred.tolist())
  return point_cloud

def plant3d_clustering():
  points_folder = os.path.join(points_parent_folder, 'simplified_points')
  output_folder = os.path.join(points_parent_folder, 'points.128.cluster.100')

  folders = os.listdir(points_folder)
  for folder in folders:
    src_folder = os.path.join(points_folder, folder)
    des_folder = os.path.join(output_folder, folder)
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)
    print('Processing: ' + des_folder)

    filenames = os.listdir(src_folder)
    point_cloud = pyoctree.Points()
    for filename in filenames:
      if filename.endswith('.points'):
        succ = point_cloud.read_points(os.path.join(src_folder, filename))
        assert succ
        point_cloud = point_cloud_clustering(point_cloud)
        succ = point_cloud.write_points(os.path.join(des_folder, filename))
        assert succ

def plant3d_generate_points_tfrecords():
  converter = '../util/convert_tfrecords.py'
  points_folder = os.path.join(points_parent_folder, 'points.128.cluster.100')

  filelist = os.path.join(points_parent_folder, 'filelist.txt')
  folders = sorted(os.listdir(points_folder))
  with open(filelist, 'w') as fid:
    for i, folder in enumerate(folders):
      filenames = os.listdir(os.path.join(points_folder, folder))
      for filename in filenames:
        if filename.endswith('.points'):
          filename = os.path.join(folder, filename)
          fid.write('%s %d\n' % (filename, i))

  tfrecords_name = os.path.join(
      points_parent_folder, 'plant3d.points.128.cluster.100.tfrecords')
  cmd = 'python %s --shuffle true --file_dir %s --list_file %s --records_name %s' % \
        (converter, points_folder, filelist, tfrecords_name)
  print(cmd)
  os.system(cmd)

#convert_to_points()
plant3d_simplify_points()
plant3d_clustering()
plant3d_generate_points_tfrecords()