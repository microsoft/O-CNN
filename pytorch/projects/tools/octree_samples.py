import torch
import ocnn


names = ['octree_%d' % i for i in range(1, 7)]
octrees = ocnn.nn.octree_samples(names)
octree  = ocnn.nn.octree_batch(octrees[:2])

for i in range(6):
  octrees[i].numpy().tofile(names[i] + '.octree')
octree.numpy().tofile('octree_batch.octree')
