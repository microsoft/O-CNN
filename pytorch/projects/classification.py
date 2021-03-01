import os
import torch
import ocnn
from tqdm import tqdm
from config import parse_args
from modelnet import ModelNet40
from torch.utils.tensorboard import SummaryWriter


def get_dataloader(flags, train=True):
  transform = ocnn.TransformCompose(flags)
  dataset = ModelNet40(flags.location, train, transform, in_memory=True)
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=flags.batch_size, shuffle=train, pin_memory=True,
      num_workers=flags.num_workers, collate_fn=ocnn.collate_octrees)
  return data_loader


def get_model(flags):
  if flags.name.lower() == 'lenet':
    model = ocnn.LeNet(flags.depth, flags.channel, flags.nout)
  elif flags.name.lower() == 'resnet':
    model = ocnn.ResNet(flags.depth, flags.channel, flags.nout, flags.resblock_num)
  else:
    raise ValueError
  return model


def train():
  model.train()

  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    # get the inputs
    octrees, labels = data[0].cuda(), data[1].cuda()

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    logits = model(octrees)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 100 == 99:
      tqdm.write('[Train iter: %5d] loss: %.3f' % (i + 1, running_loss / i))
  return running_loss / i


def test():
  model.eval()

  accuracy = 0
  for data in test_loader:
    octrees, labels = data[0].cuda(), data[1].cuda()

    with torch.no_grad():
      pred = model(octrees).max(dim=1)[1]
    accuracy += pred.eq(labels).sum().item()

  accuracy /= len(test_loader.dataset)
  tqdm.write('[Test] accuracy: %.3f' % accuracy)
  return accuracy


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
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(
      model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer, milestones=flags_solver.step_size, gamma=0.1)

  # summary
  logdir = flags_solver.logdir
  writer = SummaryWriter(logdir)
  ckpt_dir = os.path.join(logdir, 'checkpoints')
  if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
  # writer.add_graph(model, next(iter(test_loader))[0].cuda())

  # train and test
  for epoch in tqdm(range(1, flags_solver.max_iter+1), ncols=80):
    tqdm.write('[Epoch: %5d]' % epoch)
    train_loss = train()
    writer.add_scalar('train_loss', train_loss, epoch)
    if epoch % flags_solver.test_every_epoch == 0:
      test_accu = test()
      writer.add_scalar('test_accu', test_accu, epoch)
      ckpt_name = os.path.join(ckpt_dir, 'model_%05d.pth' % epoch)
      torch.save(model.state_dict(), ckpt_name)
    scheduler.step()
