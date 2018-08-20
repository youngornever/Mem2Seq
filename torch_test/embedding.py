import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import logging

logging.info('embedding test')
C = nn.Embedding(10000, 50, padding_idx=1)
C.weight.data.normal_(0, 0.1)

logging.info('test done!')