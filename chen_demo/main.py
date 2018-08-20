import numpy as np
import logging
from tqdm import tqdm
try:
    from config import *
except:
    from utils.config import *

from models.Mem2Seq import *

'''
python3 main_test.py -dec= -path= -bsz= -ds=
'''

# BLEU = False

if (args['decoder'] == "Mem2Seq"):
    if args['dataset']=='kvr':
        from utils.utils_kvr_mem2seq import *
        # BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi_mem2seq import *
    else:
        print("You need to provide the --dataset information")
else:
    if args['dataset']=='kvr':
        from utils.utils_kvr import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi import *
    else:
        print("You need to provide the --dataset information")

# Configure models
'''
path = 'save/mem2seq-BABI/5HDD12BSZ2DR0.0L1lr0.001Mem2Seq0.9776790551522375'
dir = path.split('/')
dir
['save', 'mem2seq-BABI', '5HDD12BSZ2DR0.0L1lr0.001Mem2Seq0.9776790551522375']
'''
# directory = MODEL_PATH.split("/")
# TASK = directory[2].split('HDD')[0]
# HDD = directory[2].split('HDD')[1].split('BSZ')[0]
# L = directory[2].split('L')[1].split('lr')[0]

lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']))

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](
        int(args['hidden']),max_len,max_r,lang,args['path'],args['task'], lr=0.0, n_layers=int(args['layer']), dropout=0.0, unk_mask=0)
else:
    model = globals()[args['decoder']](
        int(args['hidden']),max_len,max_r,lang,args['path'],args['task'], lr=0.0, n_layers=int(args['layer']), dropout=0.0)



