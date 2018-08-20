import torch
import torch.nn as nn
from torch.autograd import Variable

import logging

logging.info('embedding test')
scale = Variable(torch.Tensor([2]))
# scale.data[0] == 2.0
logging.info('chenxiuyi test scale')