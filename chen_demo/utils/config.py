
import os
import logging
import argparse
from tqdm import tqdm

UNK_token = 0
PAD_token = 1
EOS_token = 2
SOS_token = 3
MEM_TOKEN_SIZE = 3

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False
MAX_LENGTH = 10

PATH = '../save/mem2seq-BABI/5HDD12BSZ2DR0.0L1lr0.001Mem2Seq0.9776790551522375'

def init_params(path):
    MODEL_PATH=path
    DATASET='babi'
    DECODER='Mem2Seq'
    BSZ=1

    directory = MODEL_PATH.split("/")
    TASK = directory[2].split('HDD')[0]
    HDD = directory[2].split('HDD')[1].split('BSZ')[0]
    L = directory[2].split('L')[1].split('lr')[0]

    self = {}
    self['ds'] = self['dataset'] = DATASET
    self['t'] = self['task'] = TASK
    self['dec'] = self['decoder'] = DECODER
    self['hdd'] = self['hidden'] = HDD
    self['bsz'] = self['batch'] = BSZ
    self['lr'] = self['learn'] = None
    self['dr'] = self['drop'] = None
    self['um'] = self['unk_mask'] = True
    self['layer'] = L
    self['lm'] = self['limit'] = -10000
    self['path'] = MODEL_PATH
    self['test'] = None
    self['sample'] = None
    self['useKB'] = True
    self['ep'] = self['entPtr'] = False
    self['evalp'] = 3      # or None  ; not used
    self['an'] = self['addName'] = ''
    # parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue bAbI')
    # parser.add_argument('-ds','--dataset', help='dataset, babi or kvr', required=False)
    # parser.add_argument('-t','--task', help='Task Number', required=False)
    # parser.add_argument('-dec','--decoder', help='decoder model', required=False)
    # parser.add_argument('-hdd','--hidden', help='Hidden size', required=False)
    # parser.add_argument('-bsz','--batch', help='Batch_size', required=False)
    # parser.add_argument('-lr','--learn', help='Learning Rate', required=False)
    # parser.add_argument('-dr','--drop', help='Drop Out', required=False)
    # parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', required=False, default=1)
    # parser.add_argument('-layer','--layer', help='Layer Number', required=False)
    # parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
    # parser.add_argument('-path','--path', help='path of the file to load', required=False)
    # parser.add_argument('-test','--test', help='Testing mode', required=False)
    # parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
    # parser.add_argument('-useKB','--useKB', help='Put KB in the input or not', required=False, default=1)
    # parser.add_argument('-ep','--entPtr', help='Restrict Ptr only point to entity', required=False, default=0)
    # parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=3)
    # parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')

    return self

args = init_params(PATH)
print(args)



name = str(args['task'])+str(args['decoder'])+str(args['hidden'])+str(args['batch'])+str(args['learn'])+str(args['drop'])+str(args['layer'])+str(args['limit'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
#,filename='save/logs/{}.log'.format(str(name)))

LIMIT = int(args["limit"])
USEKB = int(args["useKB"])
ENTPTR = int(args["entPtr"])
ADDNAME = args["addName"]