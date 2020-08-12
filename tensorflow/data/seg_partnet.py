# Credit: This file is originally written by Yu-Qi Yang, modified by Peng-Shuai Wang
import os
import json
import numpy as np


current_path = os.path.dirname(os.path.realpath(__file__))
root_folder = os.path.join(current_path, '../script/dataset/partnet_segmentation')
convert_tfrecords = os.path.join(current_path, '../util/convert_tfrecords.py')

all_categoty = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher',
                'Display', 'Door', 'Earphone', 'Faucet', 'Hat', 'Keyboard',
                'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator',
                'Scissors', 'StorageFurniture', 'Table', 'TrashCan', 'Vase']
finegrained_category = ['Bed', 'Bottle', 'Chair', 'Clock', 'Dishwasher', 'Display',
                        'Door', 'Earphone', 'Faucet', 'Knife', 'Lamp', 'Microwave',
                        'Refrigerator', 'StorageFurniture', 'Table', 'TrashCan', 'Vase']
level_list_list = [[1], [1, 2, 3], [1, 3], [1], [1, 2, 3], [1, 3], [1, 2, 3],
                   [1, 3], [1, 2, 3], [1, 3], [1, 3], [1], [1], [1, 3],
                   [1, 2, 3], [1], [1, 2, 3], [1], [1, 2, 3], [1], [1, 2, 3],
                   [1, 2, 3], [1, 3], [1, 3]]
finest_level = [1, 3, 3, 1, 3, 3, 3, 3, 3, 3,
                3, 1, 1, 3, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3]
finest_level_dict = dict(zip(all_categoty, finest_level))
level_list_dict = dict(zip(all_categoty, level_list_list))


def prepare_data():
  # check the data
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  data_folder = os.path.join(root_folder, 'data_v0')
  assert os.path.exists(data_folder), \
         'Please extract the original PartNet Data to: %s' % data_folder

  # download the code
  code_folder = os.path.join(root_folder, 'partnet_dataset_master')
  if not os.path.exists(code_folder):
    url = 'https://github.com/daerduoCarey/partnet_dataset.git'
    os.system('git clone %s %s' % (url, code_folder))


def convert_ply():
  data_path = os.path.join(root_folder, 'data_v0')
  output_path = os.path.join(root_folder, 'ply')
  print('Convert the raw data to ply files ...')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  for anno_id in os.listdir(data_path):
    convert_single_data(data_path, output_path, anno_id)


def convert_points():
  input_folder = os.path.join(root_folder, 'ply')
  output_folder = os.path.join(root_folder, 'points')
  print('Convert ply files to points files ...')
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  filenames = []
  for filename in os.listdir(input_folder):
    if filename.endswith('.ply'):
      filenames.append(os.path.join(input_folder, filename))

  list_filename = os.path.join(output_folder, 'filelist_ply.txt')
  with open(list_filename, 'w') as fid:
    fid.write('\n'.join(filenames))

  cmds = ['ply2points',
          '--filenames', list_filename,
          '--output_path', output_folder,
          '--verbose', '0']
  cmd = ' '.join(cmds)
  print(cmd + '\n')
  os.system(cmd)
  os.remove(list_filename)


def convert_points_to_tfrecords():
  data_path = os.path.join(root_folder, 'points')
  split_path = os.path.join(
      root_folder, 'partnet_dataset_master/stats/train_val_test_split')
  output_path = os.path.join(root_folder, 'dataset')
  print('Convert points files to tfrecords ...')
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for category in finegrained_category:
    level_i = finest_level_dict[category]
    for phase in ['train', 'val', 'test']:
      split_json = os.path.join(split_path, "%s.%s.json" % (category, phase))
      with open(split_json, "r") as fid:
        name_json = json.load(fid)
      names = ['%s_%s_%s_level%d.points 0' % (
               category, record['model_id'], record['anno_id'], level_i)
               for record in name_json]

      filelist = "%s_%s_level%d.txt" % (category, phase, level_i)
      filelist = os.path.join(output_path, filelist)
      with open(filelist, 'w') as fid:
        fid.write('\n'.join(names))

      output_record = "%s_%s_level%d.tfrecords" % (category, phase, level_i)
      output_record = os.path.join(output_path, output_record)

      shuffle = '--shuffle true' if phase == 'train' else ''
      cmds = ['python', convert_tfrecords,
              '--file_dir', data_path,
              '--list_file', filelist,
              '--records_name', output_record,
              shuffle]

      cmd = ' '.join(cmds)
      print(cmd)
      os.system(cmd)


def convert_single_data(data_path, output_path, anno_id):
  input_path = os.path.join(data_path, anno_id)
  point_path = os.path.join(input_path, "point_sample", "sample-points-all-pts-nor-rgba-10000.txt")
  label_path = os.path.join(input_path, "point_sample", "sample-points-all-label-10000.txt")
  meta_json = os.path.join(input_path, "meta.json")
  result_json = os.path.join(input_path, "result.json")
  code_path = os.path.join(root_folder, 'partnet_dataset_master')
  part_names_dict = get_part_names(code_path, all_categoty, level_list_dict)
  node_mapping_dict = get_node_mapping(code_path, all_categoty)

  with open(meta_json, "r") as fid:
    name = json.load(fid)
  category = name["model_cat"]
  model_id = name["model_id"]
  with open(result_json, "r") as fid:
    record = json.load(fid)[0]
  new_tree_nodes = traverse(record, '', node_mapping_dict[category], [])

  points, normals, labels = read_rawdata(point_path, label_path)
  for level_i in level_list_dict[category]:
    if level_i != 3: continue  # !!! only consider the fine grained level
    new_labels = np.zeros_like(labels)
    for item in new_tree_nodes:
      if item['part_name'] in part_names_dict[category][level_i]:
        selected = np.isin(labels, item['leaf_id_list'])
        new_labels[selected] = part_names_dict[category][level_i].index(item['part_name']) + 1
    labels = new_labels

    filename = "%s_%s_%s_level%d.ply" % (category, model_id, anno_id, level_i)
    save_ply(os.path.join(output_path, filename), points, normals, labels)


def read_rawdata(point_path, label_path):
  data = np.loadtxt(point_path, dtype=np.float32)
  points, normals = data[:, 0:3], data[:, 3:6]
  labels = np.loadtxt(label_path, dtype=np.float32)
  return points, normals, labels


def get_node_mapping(dir_path, category_list):
  node_mapping_dict = {}
  cate_num = len(category_list)
  for i in range(cate_num):
    trans_fn = '%s/stats/merging_hierarchy_mapping/%s.txt' % (
        dir_path, category_list[i])
    with open(trans_fn, 'r') as fin:
      node_mapping = {item.rstrip().split()[0]: item.rstrip().split()[
          1] for item in fin.readlines()}
      node_mapping_dict[category_list[i]] = node_mapping
  return node_mapping_dict


def get_part_names(dir_path, category_list, level_list_dict):
  cate_num = len(category_list)
  part_names_dict = {}
  for i in range(cate_num):
    part_names_dict[category_list[i]] = {}
    for level_i in level_list_dict[category_list[i]]:
      stat_in_fn = '%s/stats/after_merging_label_ids/%s-level-%d.txt' % (
          dir_path, category_list[i], level_i)
      with open(stat_in_fn, 'r') as fin:
        part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
      part_names_dict[category_list[i]][level_i] = part_name_list
  return part_names_dict


def get_all_leaf_ids(record):
  if 'children' in record.keys():
    out = []
    for item in record['children']:
      out += get_all_leaf_ids(item)
    return out
  elif 'objs' in record.keys():
    return [record['id']]


def traverse(record, cur_name, node_mapping, new_result):
  if len(cur_name) == 0:
    cur_name = record['name']
  else:
    cur_name = cur_name + '/' + record['name']
  if cur_name in node_mapping.keys():
    new_part_name = node_mapping[cur_name]
    leaf_id_list = get_all_leaf_ids(record)
    new_result.append({'leaf_id_list': leaf_id_list,
                       'part_name': new_part_name})
  if 'children' in record.keys():
    for item in record['children']:
      traverse(item, cur_name, node_mapping, new_result)
  return new_result


def save_ply(filename, points, normals, labels, pts_num=10000):
  points = points.reshape((pts_num, 3))
  normals = normals.reshape((pts_num, 3))
  labels = labels.reshape((pts_num, 1))
  data = np.concatenate([points, normals, labels], axis=1)
  assert data.shape[0] == pts_num

  header = "ply\nformat ascii 1.0\nelement vertex %d\n" \
      "property float x\nproperty float y\nproperty float z\n" \
      "property float nx\nproperty float ny\nproperty float nz\n" \
      "property float label\nelement face 0\n" \
      "property list uchar int vertex_indices\nend_header\n"
  with open(filename, 'w') as fid:
    fid.write(header % pts_num)
    np.savetxt(fid, data, fmt='%.6f')


if __name__ == '__main__':
  prepare_data()
  convert_ply()
  convert_points()
  convert_points_to_tfrecords()
