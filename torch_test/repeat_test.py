import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import math
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
# ,filename='save/logs/{}.log'.format(str(name)))

logging.info('repeat test')
hidden_size = 128
v = nn.Parameter(torch.rand(hidden_size))
stdv = 1. / math.sqrt(v.size(0))
v.data.normal_(mean=0, std=stdv)

v = v.repeat(100,1).unsqueeze(1)    # (128,) ---> (100,128) ---> (100,1,128)

logging.info('test done!')
