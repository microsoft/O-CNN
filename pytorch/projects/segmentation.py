import os
import ocnn
import torch
import numpy as np

from solver import Solver, Dataset, parse_args, get_config
from datasets import get_scannet_dataset


def loss_function(logit, label):
  criterion = torch.nn.CrossEntropyLoss()
  loss = criterion(logit, label.long())
  return loss


def accuracy(logit, label):
  pred = logit.argmax(dim=1)
  accu = pred.eq(label).float().mean()
  return accu


def IoU_per_shape(logit, label, class_num):
  pred = logit.argmax(dim=1)

  IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
  intsc, union = [None] * class_num, [None] * class_num
  for k in range(class_num):
    pk, lk = pred.eq(k), label.eq(k)
    intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
    union[k] = torch.sum(torch.logical_or(pk, lk).float())

    valid = torch.sum(lk.any()) > 0
    valid_part_num += valid.item()
    IoU += valid * intsc[k] / (union[k] + esp)

  # Calculate the shape IoU for ShapeNet
  IoU /= valid_part_num + esp
  return IoU, intsc, union


class SegSolver(Solver):
  def get_model(self, flags):
    if flags.name.lower() == 'segnet':
      model = ocnn.SegNet(flags.depth, flags.channel, flags.nout, flags.interp)
    elif flags.name.lower() == 'unet':
      model = ocnn.UNet(flags.depth, flags.channel, flags.nout, flags.nempty,
                        flags.interp, flags.use_checkpoint)
    else:
      raise ValueError
    return model

  def get_dataset(self, flags):
    if flags.name.lower() == 'scannet':
      return get_scannet_dataset(flags)
    else:
      transform = ocnn.TransformCompose(flags)
      dataset = Dataset(flags.location, flags.filelist, transform, 
                        in_memory=flags.in_memory)
      return dataset, ocnn.collate_octrees

  def parse_batch(self, batch):
    octree = batch['octree'].cuda()
    if self.FLAGS.LOSS.point_wise:
      points = batch['points']
      pts = ocnn.points_batch_property(points, 'xyzi').cuda()
      label = ocnn.points_batch_property(points, 'label').squeeze().cuda()
    else:
      pts = None
      label = ocnn.octree_property(octree, 'label', self.FLAGS.MODEL.depth)
      if self.FLAGS.MODEL.nempty:
        child = ocnn.octree_property(octree, 'child', self.FLAGS.MODEL.depth)
        label = label[child >= 0]
    return octree, pts, label

  def model_forward(self, batch):
    octree, pts, label = self.parse_batch(batch)
    logit = self.model(octree, pts)
    label_mask = label > self.FLAGS.LOSS.mask  # filter labels
    return logit[label_mask], label[label_mask]

  def train_step(self, batch):
    logit, label = self.model_forward(batch)
    loss = loss_function(logit, label)
    return {'train/loss': loss}

  def test_step(self, batch):
    logit, label = self.model_forward(batch)
    if logit.shape[0] == 0:
      return None  # degenerated case
    loss = loss_function(logit, label)
    accu = accuracy(logit, label)
    num_class = self.FLAGS.LOSS.num_class
    IoU, insc, union = IoU_per_shape(logit, label, num_class)

    names = ['test/loss', 'test/accu', 'test/mIoU'] + \
            ['test/intsc_%d' % i for i in range(num_class)] + \
            ['test/union_%d' % i for i in range(num_class)]
    tensors = [loss, accu, IoU] + insc + union
    return dict(zip(names, tensors))

  def eval_step(self, batch):
    octree = batch['octree'].cuda()
    pts = ocnn.points_batch_property(batch['points'], 'xyzi').cuda()
    logit = self.model(octree, pts)
    prob = torch.nn.functional.softmax(logit, dim=1)
    label = prob.argmax(dim=1)

    assert len(batch['inbox_mask']) == 1, 'The batch_size must be 1'
    filename = '%02d.%04d.npz' % (batch['epoch'], batch['iter_num'])
    np.savez(os.path.join(self.logdir, filename), 
             prob=prob.cpu().numpy(),
             label=label.cpu().numpy(),
             inbox_mask=batch['inbox_mask'][0].numpy().astype(bool))

  def result_callback(self, avg_tracker, epoch):
    ''' Calculate the part mIoU for PartNet and ScanNet'''
    avg = avg_tracker.average()

    iou_part = 0.0
    # Labels smaller than mask is ignored. The points with the label 0 in
    # PartNet are background points, i.e., unlabeled points
    mask = self.FLAGS.LOSS.mask + 1
    num_class = self.FLAGS.LOSS.num_class
    for i in range(mask, num_class):
      instc_i = avg['test/intsc_%d' % i]
      union_i = avg['test/union_%d' % i]
      iou_part += instc_i / (union_i + 1.0e-10)
    iou_part = iou_part / (num_class - mask)
    if self.summry_writer:
      self.summry_writer.add_scalar('test/mIoU_part', iou_part, epoch)
    else:
      print('Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))


def main(TheSolver):
  get_config().LOSS.mask = -1           # mask the invalid labels
  get_config().LOSS.point_wise = False  # point-wise loss or voxel-wise loss

  FLAGS = parse_args()
  Solver.main(FLAGS, TheSolver)


if __name__ == "__main__":
  main(SegSolver)
