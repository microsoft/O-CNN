import ocnn
import torch
from tqdm import tqdm
from config import parse_args
from modelnet import ModelNet40


def get_dataloader(flags, train=True):
  transform = ocnn.TransformCompose(flags)
  dataset = ModelNet40(flags.location, train, transform, in_memory=False)
  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=flags.batch_size, shuffle=train, pin_memory=True,
      num_workers=flags.num_workers, collate_fn=ocnn.collate_octrees)
  return data_loader


def train():
  model.train()

  running_loss = 0.0
  for i, data in tqdm(enumerate(train_loader, 0)):
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
      print('[Train iter: %5d] loss: %.3f' % (i + 1, running_loss / 1000))
      running_loss = 0.0


def test():
  model.eval()

  accuracy = 0
  for data in test_loader:
    octrees, labels = data[0].cuda(), data[1].cuda()

    with torch.no_grad():
      pred = model(octrees).max(dim=1)[1]
    accuracy += pred.eq(labels).sum().item()

  accuracy /= len(test_loader.dataset)
  print('[Test] accuracy: %.3f' % accuracy)


if __name__ == "__main__":
  # configs
  FLAGS = parse_args()

  # data
  train_loader = get_dataloader(FLAGS.DATA.train, train=True)
  test_loader = get_dataloader(FLAGS.DATA.test,  train=False)

  # model
  flags_model = FLAGS.MODEL
  model = ocnn.LeNet(flags_model.depth, flags_model.channel, flags_model.nout)
  model.cuda()
  print(model)

  # loss and optimizer
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  scheduler  =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

  # train and test
  for epoch in range(1, 201):
    print('[Epoch: %5d]' % epoch)
    train()
    test()
    scheduler.step()
