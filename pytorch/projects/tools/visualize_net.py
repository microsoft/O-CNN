import ocnn
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/resnet')
octree = ocnn.octree_batch(ocnn.octree_samples(['octree_1', 'octree_2']))
model = ocnn.ResNet(depth=5, channel_in=3, nout=4, resblk_num=2)
print(model)

octree = octree.cuda()
model = model.cuda()
writer.add_graph(model, octree)
writer.flush()
