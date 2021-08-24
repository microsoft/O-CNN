import os
import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.multiprocessing
import torch.utils.data
import warnings
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .sampler import InfSampler, DistributedInfSampler


warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")


class AverageTracker:
  def __init__(self):
    self.value = None
    self.num = 0.0
    self.max_len = 70

  def update(self, value):
    if not value:
      return    # empty input, return

    value = {key: val.detach() for key, val in value.items()}
    if self.value is None:
      self.value = value
    else:
      for key, val in value.items():
        self.value[key] += val
    self.num += 1

  def average(self):
    return {key: val.item()/self.num for key, val in self.value.items()}

  @torch.no_grad()
  def average_all_gather(self):
    for key, tensor in self.value.items():
      tensors_gather = [torch.ones_like(tensor)
                        for _ in range(torch.distributed.get_world_size())]
      torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
      tensors = torch.stack(tensors_gather, dim=0)
      self.value[key] = torch.mean(tensors)

  def log(self, epoch, summry_writer=None, log_file=None):
    if not self.value:
      return  # empty, return

    avg = self.average()
    msg = 'Epoch: %d' % epoch
    for key, val in avg.items():
      msg += ', %s: %.3f' % (key, val)
      if summry_writer:
        summry_writer.add_scalar(key, val, epoch)

    if log_file:
      with open(log_file, 'a') as fid:
        fid.write(msg + '\n')
    if len(msg) > self.max_len:
      msg = msg[:self.max_len] + ' ...'
    tqdm.write(msg)


class Solver:
  def __init__(self, FLAGS, is_master=True):
    self.FLAGS = FLAGS
    self.is_master = is_master
    self.world_size = len(FLAGS.SOLVER.gpu)
    self.device = torch.cuda.current_device()
    self.disable_tqdm = not (is_master and FLAGS.SOLVER.progress_bar)
    self.start_epoch = 1

    self.model = None          # torch.nn.Module
    self.optimizer = None      # torch.optim.Optimizer
    self.scheduler = None      # torch.optim.lr_scheduler._LRScheduler
    self.summry_writer = None  # torch.utils.tensorboard.SummaryWriter
    self.log_file = None       # str, used to save training logs

  def get_model(self):
    raise NotImplementedError

  def get_dataset(self, flags):
    raise NotImplementedError

  def train_step(self, batch):
    raise NotImplementedError

  def test_step(self, batch):
    raise NotImplementedError

  def eval_step(self, batch):
    raise NotImplementedError

  def result_callback(self, avg_tracker: AverageTracker, epoch):
    pass  # additional operations based on the avg_tracker

  def config_dataloader(self, disable_train_data=False):
    flags_train, flags_test = self.FLAGS.DATA.train, self.FLAGS.DATA.test

    if not disable_train_data and not flags_train.disable:
      self.train_loader = self.get_dataloader(flags_train)
      self.train_iter = iter(self.train_loader)

    if not flags_test.disable:
      self.test_loader = self.get_dataloader(flags_test)
      self.test_iter = iter(self.test_loader)

  def get_dataloader(self, flags):
    dataset, collate_fn = self.get_dataset(flags)

    if self.world_size > 1:
      sampler = DistributedInfSampler(dataset, shuffle=flags.shuffle)
    else:
      sampler = InfSampler(dataset, shuffle=flags.shuffle)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
        sampler=sampler, collate_fn=collate_fn, pin_memory=True)
    return data_loader

  def config_model(self):
    model = self.get_model(self.FLAGS.MODEL)
    model.cuda(device=self.device)
    if self.world_size > 1:
      if self.FLAGS.MODEL.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
      model = torch.nn.parallel.DistributedDataParallel(
          module=model, device_ids=[self.device],
          output_device=self.device, broadcast_buffers=False,
          find_unused_parameters=False)
    if self.is_master:
      print(model)
    self.model = model

  def configure_optimizer(self):
    flags = self.FLAGS.SOLVER
    # The learning rate scales with regard to the world_size
    lr = flags.lr * self.world_size
    if flags.type == 'sgd':
      self.optimizer = torch.optim.SGD(
          self.model.parameters(), lr=lr, weight_decay=flags.weight_decay,
          momentum=0.9)
    elif flags.type == 'adam':
      self.optimizer = torch.optim.Adam(
          self.model.parameters(), lr=lr, weight_decay=flags.weight_decay)
    else:
      raise ValueError

    if flags.lr_type == 'step':
      self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
          self.optimizer, milestones=flags.step_size, gamma=0.1)
    elif flags.lr_type == 'cos':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          self.optimizer, flags.max_epoch, eta_min=0.001)
    elif flags.lr_type == 'poly':
      def poly(epoch): return (1 - epoch / flags.max_epoch) ** flags.lr_power
      self.scheduler = torch.optim.lr_scheduler.LambdaLR(
          self.optimizer, poly)
    elif flags.lr_type == 'constant':
      self.scheduler = torch.optim.lr_scheduler.LambdaLR(
        self.optimizer, lambda epoch: 1)
    else:
      raise ValueError

  def configure_log(self, set_writer=True):
    self.logdir = self.FLAGS.SOLVER.logdir
    self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
    self.log_file = os.path.join(self.logdir, 'log.csv')

    if self.is_master:
      tqdm.write('Logdir: ' + self.logdir)

    if self.is_master and set_writer:
      self.summry_writer = SummaryWriter(self.logdir, flush_secs=20)
      if not os.path.exists(self.ckpt_dir):
        os.makedirs(self.ckpt_dir)

  def train_epoch(self, epoch):
    self.model.train()
    if self.world_size > 1:
      self.train_loader.sampler.set_epoch(epoch)

    train_tracker = AverageTracker()
    rng = range(len(self.train_loader))
    for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
      self.optimizer.zero_grad()

      # forward
      batch = self.train_iter.next()
      batch['iter_num'] = it
      batch['epoch'] = epoch
      output = self.train_step(batch)

      # backward
      output['train/loss'].backward()
      self.optimizer.step()

      # track the averaged tensors
      train_tracker.update(output)

    # save logs
    if self.world_size > 1:
      train_tracker.average_all_gather()
    if self.is_master:
      train_tracker.log(epoch, self.summry_writer)

  def test_epoch(self, epoch):
    self.model.eval()
    test_tracker = AverageTracker()
    rng = range(len(self.test_loader))
    for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
      # forward
      batch = self.test_iter.next()
      batch['iter_num'] = it
      batch['epoch'] = epoch
      with torch.no_grad():
        output = self.test_step(batch)

      # track the averaged tensors
      test_tracker.update(output)

    if self.world_size > 1:
      test_tracker.average_all_gather()
    if self.is_master:
      test_tracker.log(epoch, self.summry_writer, self.log_file)
      self.result_callback(test_tracker, epoch)

  def eval_epoch(self, epoch):
    self.model.eval()
    for it in tqdm(range(len(self.test_loader)), ncols=80, leave=False):
      batch = self.test_iter.next()
      batch['iter_num'] = it
      batch['epoch'] = epoch
      with torch.no_grad():
        self.eval_step(batch)

  def save_checkpoint(self, epoch):
    if not self.is_master:
      return

    # clean up
    ckpts = sorted(os.listdir(self.ckpt_dir))
    ckpts = [ck for ck in ckpts if ck.endswith('.pth') or ck.endswith('.tar')]
    if len(ckpts) > self.FLAGS.SOLVER.ckpt_num:
      for ckpt in ckpts[:-self.FLAGS.SOLVER.ckpt_num]:
        os.remove(os.path.join(self.ckpt_dir, ckpt))

    # save ckpt
    model_dict = self.model.module.state_dict() \
                 if self.world_size > 1 else self.model.state_dict()
    ckpt_name = os.path.join(self.ckpt_dir, '%05d' % epoch)
    torch.save(model_dict, ckpt_name + '.model.pth')
    torch.save({'model_dict': model_dict, 'epoch': epoch,
                'optimizer_dict': self.optimizer.state_dict(),
                'scheduler_dict': self.scheduler.state_dict()},
               ckpt_name + '.solver.tar')

  def load_checkpoint(self):
    ckpt = self.FLAGS.SOLVER.ckpt
    if not ckpt:
      # If ckpt is empty, then get the latest checkpoint from ckpt_dir
      if not os.path.exists(self.ckpt_dir):  return
      ckpts = sorted(os.listdir(self.ckpt_dir))
      ckpts = [ck for ck in ckpts if ck.endswith('solver.tar')]
      if len(ckpts) > 0:
        ckpt = os.path.join(self.ckpt_dir, ckpts[-1])
    if not ckpt:  return  # return if ckpt is still empty

    # load trained model
    # check: map_location = {'cuda:0' : 'cuda:%d' % self.rank}
    trained_dict = torch.load(ckpt, map_location='cuda')
    if ckpt.endswith('.solver.tar'):
      model_dict = trained_dict['model_dict']
      self.start_epoch = trained_dict['epoch'] + 1  # !!! add 1
      if self.optimizer:
        self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
      if self.scheduler:
        self.scheduler.load_state_dict(trained_dict['scheduler_dict'])
    else:
      model_dict = trained_dict
    model = self.model.module if self.world_size > 1 else self.model
    model.load_state_dict(model_dict)

    # print messages
    if self.is_master:
      tqdm.write('Load the checkpoint: %s' % ckpt)
      tqdm.write('The start_epoch is %d' % self.start_epoch)

  def train(self):
    self.config_model()
    self.config_dataloader()
    self.configure_optimizer()
    self.configure_log()
    self.load_checkpoint()

    rng = range(self.start_epoch, self.FLAGS.SOLVER.max_epoch+1)
    for epoch in tqdm(rng, ncols=80, disable=self.disable_tqdm):
      # training epoch
      self.train_epoch(epoch)

      # update learning rate
      self.scheduler.step()
      if self.is_master:
        lr = self.scheduler.get_last_lr()  # lr is a list
        self.summry_writer.add_scalar('train/lr', lr[0], epoch)

      # testing or not
      if epoch % self.FLAGS.SOLVER.test_every_epoch != 0:
        continue

      # testing epoch
      self.test_epoch(epoch)

      # checkpoint
      self.save_checkpoint(epoch)

    # sync and exit
    if self.world_size > 1:
      torch.distributed.barrier()

  def test(self):
    self.config_model()
    self.configure_log(set_writer=False)
    self.config_dataloader(disable_train_data=True)
    self.load_checkpoint()
    self.test_epoch(epoch=0)

  def evaluate(self):
    self.config_model()
    self.configure_log(set_writer=False)
    self.config_dataloader(disable_train_data=True)
    self.load_checkpoint()
    for epoch in tqdm(range(self.FLAGS.SOLVER.eval_epoch), ncols=80):
      self.eval_epoch(epoch)

  def profile(self):
    ''' Set `DATA.train.num_workers 0` when using this function'''
    self.config_model()
    self.config_dataloader()

    # warm up
    batch = next(iter(self.train_loader))
    for _ in range(3):
      output = self.train_step(batch)
      output['train/loss'].backward()

    # profile
    with torch.autograd.profiler.profile(
            use_cuda=True, profile_memory=True,
            with_stack=True, record_shapes=True) as prof:
      output = self.train_step(batch)
      output['train/loss'].backward()

    json = os.path.join(self.FLAGS.SOLVER.logdir, 'trace.json')
    print('Save the profile into: ' + json)
    prof.export_chrome_trace(json)
    print(prof.key_averages(group_by_stack_n=10)
              .table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages(group_by_stack_n=10)
              .table(sort_by="cuda_memory_usage", row_limit=10))

  def run(self):
    eval('self.%s()' % self.FLAGS.SOLVER.run)

  @staticmethod
  def main_worker(gpu, FLAGS, TheSolver):
    world_size = len(FLAGS.SOLVER.gpu)
    if world_size > 1:
      # Set the GPU to use.
      torch.cuda.set_device(gpu)
      # Initialize the process group. Currently, the code only supports the
      # `single node + multiple GPU` mode, so the rank is equal to gpu id.
      torch.distributed.init_process_group(
          backend='nccl', init_method=FLAGS.SOLVER.dist_url,
          world_size=world_size, rank=gpu)
      # Master process is responsible for logging, writing and loading
      # checkpoints. In the multi GPU setting, we assign the master role to the
      # rank 0 process.
      is_master = gpu == 0
      solver = TheSolver(FLAGS, is_master)
    else:
      solver = TheSolver(FLAGS, is_master=True)
    solver.run()

  @staticmethod
  def main(FLAGS, TheSolver):
    num_gpus = len(FLAGS.SOLVER.gpu)
    if num_gpus > 1:
      torch.multiprocessing.spawn(
          Solver.main_worker, nprocs=num_gpus, args=(FLAGS, TheSolver))
    else:
      Solver.main_worker(0, FLAGS, TheSolver)
