import tensorflow as tf
from network_cls import network_ocnn, network_resnet


def cls_network(octree, flags, training, reuse=False):
  if flags.name.lower() == 'ocnn':
    return network_ocnn(octree, flags, training, reuse)
  elif flags.name.lower() == 'resnet':
    return network_resnet(octree, flags, training, reuse)
  else:
    print('Error, no network: ' + flags.name)



    