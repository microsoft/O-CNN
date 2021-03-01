import os
import json

current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(current_path, 'script/dataset/shapenet_segmentation')
ply2points = 'ply2points'
convert_tfrecords = os.path.join(current_path, 'util/convert_tfrecords.py')
zip_name = 'shapenetcore_partanno_segmentation_benchmark_v0_normal'

txt_folder = os.path.join(root_folder, zip_name)
ply_folder = os.path.join(root_folder, 'ply')
points_folder = os.path.join(root_folder, 'points')
dataset_folder = os.path.join(root_folder, 'datasets')

categories= ['02691156', '02773838', '02954340', '02958343',
             '03001627', '03261776', '03467517', '03624134',
             '03636649', '03642806', '03790512', '03797390',
             '03948459', '04099429', '04225987', '04379243']
names     = ['Aero',     'Bag',      'Cap',      'Car',
             'Chair',    'EarPhone', 'Guitar',   'Knife',
             'Lamp',     'Laptop',   'Motor',    'Mug',
             'Pistol',   'Rocket',   'Skate',    'Table']
seg_num   = [4, 2, 2, 4,  4,  3,  3,  2,  4,  2,  6,  2,  3,  3,  3, 3 ]
dis       = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def download_and_unzip():
  print('Downloading and unzipping ...')
  if not os.path.exists(root_folder): os.makedirs(root_folder)
  # url = 'https://shapenet.cs.stanford.edu/media/%s.zip' % zip_name
  url = 'https://www.dropbox.com/s/guy440yysyo0vrr/%s.zip?dl=0' % zip_name
  filename = os.path.join(root_folder, zip_name + '.zip')
  os.system('wget %s -O %s --no-check-certificate' % (url, filename))
  os.system('unzip %s.zip -d %s' % (filename, root_folder))

def txt_to_ply():
  print('Convert txt files to ply files ...')
  header = 'ply\nformat ascii 1.0\nelement vertex %d\n' + \
           'property float x\nproperty float y\nproperty float z\n' + \
           'property float nx\nproperty float ny\nproperty float nz\n' + \
           'property float label\nelement face 0\n' + \
           'property list uchar int vertex_indices\nend_header'
  for i, c in enumerate(categories):
    src_folder = os.path.join(txt_folder, c)
    des_folder = os.path.join(ply_folder, c)
    if not os.path.exists(des_folder): os.makedirs(des_folder)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      filename_txt = os.path.join(src_folder, filename)
      filename_ply = os.path.join(des_folder, filename[:-4] + '.ply')
      with open(filename_txt, 'r') as fid:
        lines = []
        for line in fid:
          if line == '\n': continue
          nums = line.split()
          nums[-1] = str(float(nums[-1]) - dis[i])
          lines.append(' '.join(nums))

      ply_header = header % len(lines)
      ply_content = '\n'.join([ply_header] + lines)
      with open(filename_ply, 'w') as fid:
        fid.write(ply_content)

def ply_to_points():
  print('Convert ply files to points files ...')
  for c in categories:
    src_folder = os.path.join(ply_folder, c)
    des_folder = os.path.join(points_folder, c)
    list_folder = os.path.join(ply_folder, 'fileliet')
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
    # print(cmd + '\n')
    os.system(cmd)

def points_to_tfrecords():
  print('Convert points files to tfrecords files ...')
  if not os.path.exists(dataset_folder): os.makedirs(dataset_folder)
  list_folder     = os.path.join(txt_folder, 'train_test_split')
  train_list_name = os.path.join(list_folder, 'shuffled_train_file_list.json')
  val_list_name   = os.path.join(list_folder, 'shuffled_val_file_list.json')
  test_list_name  = os.path.join(list_folder, 'shuffled_test_file_list.json')
  with open(train_list_name) as fid: train_list = json.load(fid)
  with open(val_list_name)   as fid: val_list   = json.load(fid)
  with open(test_list_name)  as fid: test_list  = json.load(fid)
  for i, c in enumerate(categories):
    filelist_name = os.path.join(list_folder, c + '_train_val.txt')
    filelist = ['%s.points %d' % (line[11:], i) for line in train_list if c in line] + \
               ['%s.points %d' % (line[11:], i) for line in val_list   if c in line]
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
    filelist = ['%s.points %d' % (line[11:], i) for line in test_list if c in line]
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

if __name__ == '__main__':
  download_and_unzip()
  txt_to_ply()
  ply_to_points()
  points_to_tfrecords()
