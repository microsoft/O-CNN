import ocnn
import torch
import numpy as np


depth, channel = 5, 3
octree = ocnn.octree_new(batch_size=1, channel=channel, node_dis=False)
octree = ocnn.octree_grow(octree, target_depth=1, full_octree=True)
octree = ocnn.octree_grow(octree, target_depth=2, full_octree=True)
octree_gt = ocnn.octree_samples(['octree_2'])[0].cuda()

for d in range(2, depth + 1):
  child = ocnn.octree_property(octree_gt, 'child', depth=d)
  label = (child > -1).to(torch.int32)
  octree = ocnn.octree_update(octree, label, depth=d, split=1)
  if d < depth:
    octree = ocnn.octree_grow(octree, target_depth=d+1, full_octree=False)

feature = ocnn.octree_property(octree_gt, 'feature', depth)
octree = ocnn.octree_set_property(octree, feature, depth)

print('Please check the output files:`octree_1.octree` and `octree_2.octree`.\n'
      'The MD5 of `octree_1.octree`: FEB7C4AF43669EB0FC62632C71D1C938\n'
      'The MD5 of `octree_2.octree`: D569D5BB23D34795C5FD81397F56275B')
octree_gt.cpu().numpy().tofile('octree_1.octree')
octree.cpu().numpy().tofile('octree_2.octree')

