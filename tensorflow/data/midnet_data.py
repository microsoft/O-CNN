import os
import sys


abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'script/dataset/midnet_data')


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


if __name__ == '__main__':
  download_data()
