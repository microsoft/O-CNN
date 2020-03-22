import tensorflow as tf


class CosLR:
  def __init__(self, flags):
    self.flags = flags

  def __call__(self, global_step):
    with tf.variable_scope('cos_lr'):
      pi, mul = 3.1415926, 0.001
      step_size = self.flags.step_size[0]
      max_iter  = self.flags.max_iter * 0.9
      max_epoch = max_iter / step_size
      lr_max = self.flags.learning_rate
      lr_min = self.flags.learning_rate * mul
      epoch = tf.floordiv(tf.cast(global_step, tf.float32), step_size)
      val   = tf.minimum(epoch / max_epoch, 1.0)
      lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(pi * val))
    return lr


class StepLR:
  def __init__(self, flags):
    self.flags = flags

  def __call__(self, global_step):
    with tf.variable_scope('step_lr'):
      step_size = list(self.flags.step_size)
      for i in range(len(step_size), 5): 
        step_size.append(step_size[-1])

      steps = step_size
      for i in range(1, 5):
        steps[i] = steps[i-1] + steps[i]
      lr_values = [self.flags.gamma**i * self.flags.learning_rate for i in range(0, 6)]

      lr = tf.train.piecewise_constant(global_step, steps, lr_values)
    return lr


class LRFactory:
  def __init__(self, flags):
    self.flags = flags
    if self.flags.lr_type == 'step':
      self.lr = StepLR(flags)
    elif self.flags.lr_type == 'cos':
      self.lr = CosLR(flags)
    else:
      print('Error, unsupported learning rate: ' + self.flags.lr_type)
  
  def __call__(self, global_step):
    return self.lr(global_step)