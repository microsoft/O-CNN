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


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True,
                    help='The command to run.')
parser.add_argument('--converter', type=str, required=False,
                    default='util/convert_tfrecords.py',
                    help='The path of the convert_tfrecords')
parser.add_argument('--scanner', type=str,  required=False,
                    help='The path of the virtual_scanner')
parser.add_argument('--simplify_points', type=str, required=False,
                    default='simplify_points',
                    help='The path of the simplify_points')


abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'script/dataset/midnet_data')

args = parser.parse_args()
converter = os.path.join(abs_path, args.converter)
virtual_scanner = args.scanner
simplify = args.simplify_points


def download_data():
  # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'https://www.dropbox.com/s/lynuimh1bbtnkty/midnet_data.zip?dl=0'
  cmd = 'wget %s -O %s.zip' % (url, root_folder)
  print(cmd)
  os.system(cmd)

  # unzip
  cmd = 'unzip %s.zip -d %s/..' % (root_folder, root_folder)
  print(cmd)
  os.system(cmd)


def shapenet_unzip():
  shapenet = os.path.join(root_folder, 'ShapeNetCore.v1.zip')
  cmd = 'unzip %s -d %s' % (shapenet, root_folder)
  print(cmd)
  os.system(cmd)

  shapenet_folder = os.path.join(root_folder, 'ShapeNetCore.v1')
  filenames = os.listdir(shapenet_folder)
  for filename in filenames:
    abs_name = os.path.join(shapenet_folder, filename)
    if not filename.endswith('.zip'):
      os.remove(abs_name)
    else:
      cmd = 'unzip %s -d %s' % (abs_name, shapenet_folder)
      print(cmd)
      os.system(cmd)


def shapenet_move_objs():
  shapenet_folder = os.path.join(root_folder, 'ShapeNetCore.v1')
  mesh_folder = os.path.join(root_folder, 'mesh')
  folders = os.listdir(shapenet_folder)
  for folder in folders:
    src_folder = os.path.join(shapenet_folder, folder)
    des_folder = os.path.join(mesh_folder, folder)
    if not os.path.isdir(src_folder):
      continue
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      src_filename = os.path.join(src_folder, filename, 'model.obj')
      des_filename = os.path.join(des_folder, filename + '.obj')
      if not os.path.exists(src_filename):
        print('Warning: not exist - ', src_filename)
        continue
      os.rename(src_filename, des_filename)


def shapenet_convert_mesh_to_points():
  mesh_folder = os.path.join(root_folder, 'mesh')
  # Delete the following 3 files since the virtualscanner can not deal with them
  filelist = ['03624134/67ada28ebc79cc75a056f196c127ed77.obj',
              '04074963/b65b590a565fa2547e1c85c5c15da7fb.obj',
              '04090263/4a32519f44dc84aabafe26e2eb69ebf4.obj']
  for filename in filelist:
    filename = os.path.join(mesh_folder, filename)
    if os.path.exists(filename):
      os.remove(filename)

  # run virtualscanner
  folders = os.listdir(mesh_folder)
  for folder in folders:
    curr_folder = os.path.join(mesh_folder, folder)
    cmd = '%s %s 14' % (virtual_scanner,  curr_folder)
    print(cmd)
    os.system(cmd)

  # move points
  points_folder = os.path.join(root_folder, 'points.dense')
  for folder in folders:
    src_folder = os.path.join(mesh_folder, folder)
    des_folder = os.path.join(points_folder, folder)
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)
    filenames = os.listdir(src_folder)
    for filename in filenames:
      if filename.endswith('.points'):
        os.rename(os.path.join(src_folder, filename),
                  os.path.join(des_folder, filename))


def shapenet_simplify_points(resolution=64):
  # rename and backup the original folders
  points_folder = os.path.join(root_folder, 'points')
  original_folder = os.path.join(root_folder, 'points.dense')
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
    output_path = os.path.join(points_folder, folder)
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    cmd = '%s --filenames %s --output_path %s --dim %d' % \
          (simplify, filelist_name, output_path, resolution)
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


def shapenet_clustering():
  points_folder = os.path.join(root_folder, 'points')
  output_folder = os.path.join(root_folder, 'points.64.cluster.100')

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


def shapenet_generate_points_tfrecords():
  points_folder = os.path.join(root_folder, 'points.64.cluster.100')

  filelist = os.path.join(root_folder, 'filelist.txt')
  folders = sorted(os.listdir(points_folder))
  with open(filelist, 'w') as fid:
    for i, folder in enumerate(folders):
      filenames = os.listdir(os.path.join(points_folder, folder))
      for filename in filenames:
        if filename.endswith('.points'):
          filename = os.path.join(folder, filename)
          fid.write('%s %d\n' % (filename, i))

  tfrecords_name = os.path.join(
      root_folder, 'shapenet.points.64.cluster.100.tfrecords')
  cmd = 'python %s --shuffle true --file_dir %s --list_file %s --records_name %s' % \
        (converter, points_folder, filelist, tfrecords_name)
  print(cmd)
  os.system(cmd)


def shapenet_create_tfrecords():
  shapenet_unzip()
  shapenet_move_objs()
  shapenet_convert_mesh_to_points()
  shapenet_simplify_points()
  shapenet_clustering()
  shapenet_generate_points_tfrecords()


if __name__ == '__main__':
  eval('%s()' % args.run)
