import os
import torch
import ocnn
from tqdm import tqdm
from config import parse_args
from dataset import Dataset
from torch.utils.tensorboard import SummaryWriter


def get_dataloader(flags, train=True):
  transform = ocnn.TransformCompose(flags)
  dataset = Dataset(flags.location, flags.filelist, transform, in_memory=True)
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=flags.batch_size, shuffle=train, pin_memory=True,
      num_workers=flags.num_workers, collate_fn=ocnn.collate_octrees)
  return data_loader


def get_model(flags):
  if flags.name.lower() == 'segnet':
    model = ocnn.SegNet(flags.depth, flags.channel, flags.nout)
  else:
    raise ValueError
  return model


def train():
  model.train()

  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    # get the inputs
    octrees = data[0].cuda()
    labels = ocnn.octree_property(octrees, 'label', FLAGS.MODEL.depth)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    logits = model(octrees)
    logits = logits.squeeze().transpose(0, 1)  # N x C
    loss = loss_functions_seg(logits, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 100 == 99:
      tqdm.write('[Train iter: %5d] loss: %.3f' % (i + 1, running_loss / i))
  return running_loss / i


def test():
  model.eval()

  accu, mIoU, counter = 0, 0, 0
  for data in test_loader:
    octrees = data[0].cuda()
    labels = ocnn.octree_property(octrees, 'label', FLAGS.MODEL.depth)

    with torch.no_grad():
      logits = model(octrees)
      logits = logits.squeeze().transpose(0, 1)  # N x C

    counter += 1
    accu += accuracy(logits, labels)
    mIoU += IoU_per_shape(logits, labels, FLAGS.LOSS.num_class)

  accu /= counter
  mIoU /= counter
  tqdm.write('[Test] accuracy: %.3f, mIoU: %.3f' % (accu, mIoU))
  return accu, mIoU


def loss_functions_seg(logit, label, mask=-1):
  label_mask = label > mask  # filter label -1
  masked_logit = logit[label_mask, :]
  masked_label = label[label_mask]
  criterion = torch.nn.CrossEntropyLoss()
  loss = criterion(masked_logit, masked_label.long())
  return loss


def accuracy(logit, label, mask=-1):
  label_mask = label > mask  # filter label -1
  masked_logit = logit[label_mask, :]
  label_mask = label[label_mask]
  pred = masked_logit.argmax(dim=1)
  accu = pred.eq(label_mask).float().mean()
  return accu.item()


def IoU_per_shape(logit, label, class_num, mask=-1):
  label_mask = label > mask  # filter label -1
  masked_logit = logit[label_mask, :]
  masked_label = label[label_mask]
  pred = masked_logit.argmax(dim=1)

  IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
  for k in range(class_num):
    pk, lk = pred.eq(k), masked_label.eq(k)
    intsc = torch.sum(pk & lk)
    union = torch.sum(pk | lk)
    valid = torch.sum(lk.any()) > 0
    valid_part_num += valid.item()
    IoU += valid * intsc / (union + esp)
  IoU /= valid_part_num + esp
  return IoU.item()


if __name__ == "__main__":
  # configs
  FLAGS = parse_args()

  # data
  train_loader = get_dataloader(FLAGS.DATA.train, train=True)
  test_loader = get_dataloader(FLAGS.DATA.test,  train=False)

  # model
  model = get_model(FLAGS.MODEL)
  model.cuda()
  print(model)

  # loss and optimizer
  flags_solver = FLAGS.SOLVER
  optimizer = torch.optim.SGD(
      model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer, milestones=flags_solver.step_size, gamma=0.1)

  # summary
  logdir = flags_solver.logdir
  writer = SummaryWriter(logdir)
  ckpt_dir = os.path.join(logdir, 'checkpoints')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  # writer.add_graph(model, next(iter(test_loader))[0].cuda())

  # train and test
  for epoch in tqdm(range(1, flags_solver.max_epoch+1), ncols=80):
    tqdm.write('[Epoch: %5d]' % epoch)
    train_loss = train()
    writer.add_scalar('train_loss', train_loss, epoch)
    if epoch % flags_solver.test_every_epoch == 0:
      test_accu, test_mIoU = test()
      writer.add_scalar('test_accu', test_accu, epoch)
      writer.add_scalar('test_mIoU', test_mIoU, epoch)
      ckpt_name = os.path.join(ckpt_dir, 'model_%05d.pth' % epoch)
      torch.save(model.state_dict(), ckpt_name)
    scheduler.step()
