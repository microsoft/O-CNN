import os
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True,
                    help='The command to run.')
parser.add_argument('--scanner', type=str,  required=False,
                    help='The path of the virtual_scanner')
parser.add_argument('--simplify_points', type=str, required=False,
                    default='simplify_points',
                    help='The path of the simplify_points')
parser.add_argument('--transform_points', type=str, required=False,
                    default='transform_points',
                    help='The path of the transform_points')
parser.add_argument('--align_y', type=str, required=False, default='false',
                    help='Align the points with y axis')

abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data/ModelNet40')

args = parser.parse_args()
virtual_scanner = args.scanner
simplify = args.simplify_points
transform = args.transform_points


def download_m40():
  # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
  cmd = 'wget %s -O %s/ModelNet40.zip' % (url, root_folder)
  print(cmd)
  os.system(cmd)

  # unzip
  cmd = 'unzip %s/ModelNet40.zip -d %s' % (root_folder, root_folder)
  print(cmd)
  os.system(cmd)


def download_m40_points():
  # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'https://www.dropbox.com/s/m233s9eza3acj2a/ModelNet40.points.zip?dl=0'
  zip_file = os.path.join(root_folder, 'ModelNet40.points.zip')
  cmd = 'wget %s -O %s' % (url, zip_file)
  print(cmd)
  os.system(cmd)

  # unzip
  cmd = 'unzip %s -d %s/ModelNet40.points' % (zip_file, root_folder)
  print(cmd)
  os.system(cmd)


def clean_off_file(filename):
  # read the contents of the file
  with open(filename) as fid:
    file_str = fid.read()
  # fix the file
  if file_str[0:3] != 'OFF':
    print('Error: not an OFF file: ' + filename)
  elif file_str[0:4] != 'OFF\n':
    print('Info: fix an OFF file: ' + filename)
    new_str = file_str[0:3] + '\n' + file_str[3:]
    with open(filename, 'w') as f_rewrite:
      f_rewrite.write(new_str)


def get_filelist(root_folder, train=True, suffix='off', ratio=1.0):
  filelist, category = [], []
  folders = sorted(os.listdir(root_folder))
  assert(len(folders) == 40)
  for idx, folder in enumerate(folders):
    subfolder = 'train' if train else 'test'
    current_folder = os.path.join(root_folder, folder, subfolder)
    filenames = sorted(os.listdir(current_folder))
    filenames = [fname for fname in filenames if fname.endswith(suffix)]
    total_num = math.ceil(len(filenames) * ratio)
    for i in range(total_num):
      filelist.append(os.path.join(folder, subfolder, filenames[i]))
      category.append(idx)
  return filelist, category


def move_files(src_folder, des_folder, suffix):
  folders = os.listdir(src_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_src_folder = os.path.join(src_folder, folder, subfolder)
      curr_des_folder = os.path.join(des_folder, folder, subfolder)
      if not os.path.exists(curr_des_folder):
        os.makedirs(curr_des_folder)
      filenames = os.listdir(curr_src_folder)
      for filename in filenames:
        if filename.endswith(suffix):
          os.rename(os.path.join(curr_src_folder, filename),
                    os.path.join(curr_des_folder, filename))


def convert_mesh_to_points():
  mesh_folder = os.path.join(root_folder, 'ModelNet40')
  # Delete the following 3 files since the virtualscanner can not deal with them
  filelist = ['cone/train/cone_0117.off',
              'curtain/train/curtain_0066.off',
              'car/train/car_0021.off.off']
  for filename in filelist:
    filename = os.path.join(mesh_folder, filename)
    if os.path.exists(filename):
      os.remove(filename)

  # clean the off files
  train_list, _ = get_filelist(mesh_folder, train=True,  suffix='off')
  test_list, _ = get_filelist(mesh_folder, train=False, suffix='off')
  filelist = train_list + test_list
  for filename in filelist:
    clean_off_file(os.path.join(mesh_folder, filename))

  # run virtualscanner
  folders = os.listdir(mesh_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(mesh_folder, folder, subfolder)
      cmd = '%s %s 14' % (virtual_scanner,  curr_folder)
      print(cmd)
      os.system(cmd)

  # move points
  move_files(mesh_folder, mesh_folder + '.points', 'points')


def simplify_points(resolution=64):
  # rename and backup the original folders
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  original_folder = points_folder + ".dense"
  if os.path.exists(points_folder):
    os.rename(points_folder, original_folder)

  folders = os.listdir(original_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(original_folder, folder, subfolder)
      # write filelist to disk
      filenames = os.listdir(curr_folder)
      filelist_name = os.path.join(curr_folder, 'list.txt')
      with open(filelist_name, 'w') as fid:
        for filename in filenames:
          if filename.endswith('.points'):
            fid.write(os.path.join(curr_folder, filename) + '\n')
      # run simplify_points
      output_path = os.path.join(points_folder, folder, subfolder)
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      cmd = '%s --filenames %s --output_path %s --dim %d' % \
            (simplify, filelist_name, output_path, resolution)
      print(cmd)
      os.system(cmd)
      os.remove(filelist_name)


def transform_points():
  points_folder = os.path.join(root_folder, 'ModelNet40.points')
  output_folder = os.path.join(root_folder, 'ModelNet40.points.y')
  folders = os.listdir(points_folder)
  for folder in folders:
    for subfolder in ['train', 'test']:
      curr_folder = os.path.join(points_folder, folder, subfolder)
      output_path = os.path.join(output_folder, folder, subfolder)
      if not os.path.exists(output_path):
        os.makedirs(output_path)

      # write filelist to disk
      filenames = os.listdir(curr_folder)
      filelist_name = os.path.join(curr_folder, 'list.txt')
      with open(filelist_name, 'w') as fid:
        for filename in filenames:
          if filename.endswith('.points'):
            fid.write(os.path.join(curr_folder, filename) + '\n')

      # write the transformation matrix
      mat = '0 0 1 1 0 0 0 1 0'
      mat_name = os.path.join(curr_folder, 'mat.txt')
      with open(mat_name, 'w') as fid:
        fid.write(mat)

      # run transform points
      cmd = '%s --filenames %s --output_path %s --mat %s' % \
            (transform, filelist_name, output_path, mat_name)
      print(cmd)
      os.system(cmd)
      os.remove(filelist_name)
      os.remove(mat_name)


def generate_points_filelist():
  points_folder = os.path.join(root_folder, 'ModelNet40.points')

  for folder in ['train', 'test']:
    train = folder == 'train'
    filelist, idx = get_filelist(points_folder, train=train, suffix='points')
    prefix = 'm40_' + folder
    filename = os.path.join(root_folder, '%s_points_list.txt' % prefix)
    print('Save to %s' % filename)
    with open(filename, 'w') as fid:
      for i in range(len(filelist)):
        fid.write('%s %d\n' % (filelist[i], idx[i]))


def generate_points_filelist_ratios():
  ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
  points_folder = os.path.join(root_folder, 'ModelNet40.points.y')

  for folder in ['train', 'test']:
    train = folder == 'train'
    for ratio in ratios:
      if train == False and ratio < 1:
        continue
      prefix = 'm40_y_%.02f_%s' % (ratio, folder)
      filename = os.path.join(root_folder, '%s_points_list.txt' % prefix)
      filelist, idx = get_filelist(points_folder, train=train,
                                   suffix='points', ratio=ratio)
      print('Save to %s' % filename)
      with open(filename, 'w') as fid:
        for i in range(len(filelist)):
          fid.write('%s %d\n' % (filelist[i], idx[i]))


if __name__ == '__main__':
  eval('%s()' % args.run)
