import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
# import datetime
# from utils.measures import wer, moses_multi_bleu
# import matplotlib
# # matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
from sklearn.metrics import f1_score
try:
    from utils.utils_babi_mem2seq import generate_memory, MEM_TOKEN_SIZE
except:
    from utils_babi_mem2seq import generate_memory
from utils.config import *
from utils.utils_kb import kb_dict, api_call
import logging

class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.task = task        # used for save and get metrics
        self.input_size = lang.n_words      # not used
        self.output_size = lang.n_words     # used for generate words
        self.hidden_size = hidden_size      # ENCODER & DECODER
        self.n_layers = n_layers            # as hops
        self.max_len = max_len      # max input         ; 在处理时有+1操作     not used
        self.max_r = max_r          # max response len  ; 在处理时有+1操作     used for eval()
        self.lang = lang            # index2word; word2idx
        self.lr = lr
        self.dropout = dropout
        self.unk_mask = unk_mask    # a trick in the paper; mask input as unk
        self.History = []

        if path:    # for saved model
            if USE_CUDA:        # defined in utils.config
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderrMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,mode='max',factor=0.5,patience=1,min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr =  self.loss_ptr / self.print_every
        print_loss_vac =  self.loss_vac / self.print_every
        self.print_every += 1     
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg,print_loss_vac,print_loss_ptr)
    
    def save_model(self, dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        directory = 'save/mem2seq-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')


    def History2Context(self):
        # history is a list of list
        # self.Context = self.History
        self.Context = []
        end_time = len(self.History)
        time_counter = 1
        for ii, one_turn in enumerate(self.History):
            if len(one_turn) == 1:
                if ii == end_time-1:
                    u = one_turn[0]        # the last sent
                    sent_new = generate_memory(u, "$u", str(time_counter))
                    self.Context += sent_new
                    time_counter += 1
                else:  # this is KB
                    r = one_turn[0]  # the last sent
                    sent_new = generate_memory(r, "", "")
                    self.Context += sent_new
            else:
                u, r = one_turn
                sent_new = generate_memory(u, "$u", str(time_counter))
                self.Context += sent_new
                sent_new = generate_memory(r, "$s", str(time_counter))
                self.Context += sent_new
                time_counter += 1
        self.Context += [['$$$$']*MEM_TOKEN_SIZE]   # the end of history

    def Context2Input(self):
        def preprocess(sequence, word2id, trg=False):
            """Converts words to ids."""
            if trg:
                story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
            else:
                story = []
                # 好象是n元组
                for i, word_triple in enumerate(sequence):
                    story.append([])
                    for ii, word in enumerate(word_triple):
                        temp = word2id[word] if word in word2id else UNK_token
                        story[i].append(temp)
            story = [story]     # 加一维
            try:
                story = torch.Tensor(story)     # (1, t, 3)
            except:
                print(sequence)
                print(story)
            return story
        src_context = preprocess(self.Context, self.lang.word2index)    # (1, t, 3)
        length = [len(self.Context)]       # (1,)
        context = Variable(src_context)          # (1, T, 3)

        context = context.transpose(0,1)

        if USE_CUDA:
            context = context.cuda()

        return context, length, src_context

    def generate_res(self, batch_size, input_batches, input_lengths, src_context):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)  # equivalently, self.encoder.eval()
        self.decoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)  # (1,B,E)
        # get the embedding
        self.decoder.load_memory(input_batches.transpose(0, 1))
        # Prepare input and output variables        # (B,)
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))
        # all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            # all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()

        # Run through decoder one time step at a time
        for t in range(self.max_r):
            # (B,M)     (B,V)           (1,B,E or H)
            decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topvi = decoder_vacab.data.topk(1)  # topv, topp not used
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)
            # shape (1,B)
            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(toppi.view(1, -1)))
            next_in = [
                top_ptr_i.squeeze()[i].data[0] if (toppi.squeeze()[i] < input_lengths[i] - 1) else topvi.squeeze()[
                    i] for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # （word, $u, time）  to get the word seq
            p = []
            for elm in src_context:  # shape(B,T,3)
                elm_temp = [word_triple[0] for word_triple in elm]
                p.append(elm_temp)  # shape (B,T)

            temp = []
            from_which = []
            for i in range(batch_size):
                # 因为起始位置为0; 所以长度-1；而且最后元素为$$$$符号,所以小于
                if (toppi.squeeze()[i] < len(p[i]) - 1):  # p: shape (B,T); actually a list of list
                    temp.append(p[i][toppi.squeeze()[i]])
                    from_which.append('p')  # from ptr
                else:
                    ind = topvi.squeeze()[i]
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind])
                    from_which.append('v')  # from generated vocab
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac    # shape (T,B)

    def demo_res(self, user_crt_saying):
        self.History.append([])
        self.History[-1].append(user_crt_saying.strip())

        self.History2Context()
        input_, length_, src_ = self.Context2Input()

        # res_ is list of list; (T,B)
        res_ = generate_res(self, batch_size=1, input_batches=input_, input_lengths=length_, src_context=src_)
        res = ''
        for bbb in res_:
            for www in bbb:
                res = res + www + ' '
        res = res.strip()
        if 'api_call' in res:
            self.History[-1].append(res)  # first append res
            # TODO: retrieve
            param = res.strip().split(' ').remove('api_call')
            param = tuple(param) + (kb_dict,)
            logging.info('api_call:     ', res)
            kb_data = api_call(*param)
            self.History += kb_data
        else:
            self.History[-1].append(res)


#     def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths,
#                        target_index, target_gate, src_plain, conv_seqs, conv_lengths):
#         # Set to not-training mode to disable dropout
#         self.encoder.train(False)       # equivalently, self.encoder.eval()
#         self.decoder.train(False)
#         # Run words through encoder
#         decoder_hidden = self.encoder(input_batches).unsqueeze(0)   # (1,B,E)
#         # get the embedding
#         self.decoder.load_memory(input_batches.transpose(0,1))
#
#         # Prepare input and output variables        # (B,)
#         decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
#
#         decoded_words = []
#         all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.output_size))
#         all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))
#         # all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
#         # Move new Variables to CUDA
#
#         if USE_CUDA:
#             all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
#             all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
#             # all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
#             decoder_input = decoder_input.cuda()
#
#         # （word, $u, time）  to get the word seq
#         p = []
#         for elm in src_plain:       # shape(B,T,3)
#             elm_temp = [word_triple[0] for word_triple in elm]
#             p.append(elm_temp)      # shape (B,T)
#
#         self.from_whichs = []       # ptr or P_vocab
#         acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
#         # Run through decoder one time step at a time
#         for t in range(self.max_r):
#             # (B,M)     (B,V)           (1,B,E or H)
#             decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
#             all_decoder_outputs_vocab[t] = decoder_vacab
#             topv, topvi = decoder_vacab.data.topk(1)        # topv, topp not used
#             all_decoder_outputs_ptr[t] = decoder_ptr
#             topp, toppi = decoder_ptr.data.topk(1)
#             # shape (1,B)
#             top_ptr_i = torch.gather(input_batches[:,:,0],0,Variable(toppi.view(1, -1)))
#             next_in = [top_ptr_i.squeeze()[i].data[0] if(toppi.squeeze()[i] < input_lengths[i]-1) else topvi.squeeze()[i] for i in range(batch_size)]
#
#             decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
#             if USE_CUDA: decoder_input = decoder_input.cuda()
#
#             temp = []
#             from_which = []
#             for i in range(batch_size):
#                 # 因为起始位置为0; 所以长度-1；而且最后元素为$$$$符号,所以小于
#                 if(toppi.squeeze()[i] < len(p[i])-1):       # p: shape (B,T); actually a list of list
#                     temp.append(p[i][toppi.squeeze()[i]])
#                     from_which.append('p')      # from ptr
#                 else:
#                     ind = topvi.squeeze()[i]
#                     if ind == EOS_token:
#                         temp.append('<EOS>')
#                     else:
#                         temp.append(self.lang.index2word[ind])
#                     from_which.append('v')      # from generated vocab
#             decoded_words.append(temp)
#             self.from_whichs.append(from_which)
#         self.from_whichs = np.array(self.from_whichs)
#
#         # indices = torch.LongTensor(range(target_gate.size(0)))
#         # if USE_CUDA: indices = indices.cuda()
#
#         # ## acc pointer
#         # y_ptr_hat = all_decoder_outputs_ptr.topk(1)[1].squeeze()
#         # y_ptr_hat = torch.index_select(y_ptr_hat, 0, indices)
#         # y_ptr = target_index
#         # acc_ptr = y_ptr.eq(y_ptr_hat).sum()
#         # acc_ptr = acc_ptr.data[0]/(y_ptr_hat.size(0)*y_ptr_hat.size(1))
#         # ## acc vocab
#         # y_vac_hat = all_decoder_outputs_vocab.topk(1)[1].squeeze()
#         # y_vac_hat = torch.index_select(y_vac_hat, 0, indices)
#         # y_vac = target_batches
#         # acc_vac = y_vac.eq(y_vac_hat).sum()
#         # acc_vac = acc_vac.data[0]/(y_vac_hat.size(0)*y_vac_hat.size(1))
#
#         # Set back to training mode
#         self.encoder.train(True)
#         self.decoder.train(True)
#         return decoded_words    # , acc_ptr, acc_vac    # shape (T,B)
#
#     def evaluate(self, dev, avg_best, BLEU=False):
#         logging.info("STARTING EVALUATION")
#         acc_avg = 0.0
#         wer_avg = 0.0
#         bleu_avg = 0.0
#         acc_P = 0.0
#         acc_V = 0.0
#         microF1_PRED,microF1_PRED_cal,microF1_PRED_nav,microF1_PRED_wet = [],[],[],[]
#         microF1_TRUE,microF1_TRUE_cal,microF1_TRUE_nav,microF1_TRUE_wet = [],[],[],[]
#         # 在whole eval_dataset上计算的
#         ref = []
#         hyp = []
#         ref_s = ""
#         hyp_s = ""
#         dialog_acc_dict = {}    # 在whole eval_dataset上计算的
#         pbar = tqdm(enumerate(dev), total=len(dev))
#         for j, data_dev in pbar:
#             if args['dataset']=='kvr':
#                 '''
#                 batch_size,
#                 input_batches, input_lengths,
#                 target_batches, target_lengths,
#                 target_index,target_gate,
#                 src_plain,
#                 conv_seqs, conv_lengths'''
#                 # output shape (T,B)
#                 words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],
#                                     data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6], data_dev[-2], data_dev[-1])
#             else:
#                 words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],
#                         data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6], data_dev[-4], data_dev[-3])
#             # acc_P += acc_ptr
#             # acc_V += acc_vac
#             acc = 0     # 在one batch里计算的
#             w = 0
#             temp_gen = []
#
#             # Permute the dimensions of an array
#             for i, row in enumerate(np.transpose(words)):   # shape (B,T)
#                 st = ''
#                 for e in row:
#                     if e== '<EOS>': break
#                     else: st += e + ' '
#                 temp_gen.append(st)
#                 # data_dev[7] may be the correct sentences; shape(B,T)
#                 correct = data_dev[7][i]    # this is response sentences
#                 # compute F1 SCORE
#                 if args['dataset']=='kvr':
#                     f1_true,f1_pred = computeF1(data_dev[8][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
#                     microF1_TRUE += f1_true
#                     microF1_PRED += f1_pred     # 全部是1,估计用来做分母的,多余的？？
#                     f1_true,f1_pred = computeF1(data_dev[9][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
#                     microF1_TRUE_cal += f1_true
#                     microF1_PRED_cal += f1_pred
#                     f1_true,f1_pred = computeF1(data_dev[10][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
#                     microF1_TRUE_nav += f1_true
#                     microF1_PRED_nav += f1_pred
#                     f1_true,f1_pred = computeF1(data_dev[11][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
#                     microF1_TRUE_wet += f1_true
#                     microF1_PRED_wet += f1_pred
#                 elif args['dataset']=='babi' and int(self.task)==6:
#                     f1_true, f1_pred = computeF1(data_dev[-2][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
#                     microF1_TRUE += f1_true
#                     microF1_PRED += f1_pred
#
#                 if args['dataset']=='babi':
#                     # ID
#                     if data_dev[-1][i] not in dialog_acc_dict.keys():
#                         dialog_acc_dict[data_dev[-1][i]] = []
#                     if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
#                         acc += 1    # 在one batch里计算的
#                         dialog_acc_dict[data_dev[-1][i]].append(1)
#                     else:           # 在whole eval_dataset上计算的
#                         dialog_acc_dict[data_dev[-1][i]].append(0)
#                 else:
#                     if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
#                         acc += 1
#                 #    print("Correct:"+str(correct.lstrip().rstrip()))
#                 #    print("\tPredict:"+str(st.lstrip().rstrip()))
#                 #    print("\tFrom:"+str(self.from_whichs[:,i]))
#
#                 w += wer(correct.lstrip().rstrip(), st.lstrip().rstrip())
#                 ref.append(str(correct.lstrip().rstrip()))
#                 hyp.append(str(st.lstrip().rstrip()))
#                 ref_s += str(correct.lstrip().rstrip()) + "\n"
#                 hyp_s += str(st.lstrip().rstrip()) + "\n"
#
#             acc_avg += acc/float(len(data_dev[1]))    # len(data_dev[1]) = batch_size
#             wer_avg += w/float(len(data_dev[1]))      # len(dev) = num of batches
#             # TODO: 有点不合理啊; 除以j应该比较合理;
#             pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),
#                                                                     wer_avg/float(len(dev))))
#
#         # dialog accuracy
#         if args['dataset']=='babi':
#             dia_acc = 0
#             for k in dialog_acc_dict.keys():
#                 if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
#                     dia_acc += 1
#             logging.info("Dialog Accuracy:\t"+str(dia_acc*1.0/len(dialog_acc_dict.keys())))
#
#         if args['dataset']=='kvr':
#             logging.info("F1 SCORE:\t"+str(f1_score(microF1_TRUE, microF1_PRED, average='micro')))
#             logging.info("F1 CAL:\t"+str(f1_score(microF1_TRUE_cal, microF1_PRED_cal, average='micro')))
#             logging.info("F1 WET:\t"+str(f1_score(microF1_TRUE_wet, microF1_PRED_wet, average='micro')))
#             logging.info("F1 NAV:\t"+str(f1_score(microF1_TRUE_nav, microF1_PRED_nav, average='micro')))
#         elif args['dataset']=='babi' and int(self.task)==6 :
#             logging.info("F1 SCORE:\t"+str(f1_score(microF1_TRUE, microF1_PRED, average='micro')))
#
#
#         bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
#         logging.info("BLEU SCORE:"+str(bleu_score))
#         if (BLEU):
#             if (bleu_score >= avg_best):
#                 self.save_model(str(self.name)+str(bleu_score))
#                 logging.info("MODEL SAVED")
#             return bleu_score
#         else:
#             acc_avg = acc_avg/float(len(dev))
#             if (acc_avg >= avg_best):
#                 self.save_model(str(self.name)+str(acc_avg))
#                 logging.info("MODEL SAVED")
#             return acc_avg
#
#
# def computeF1(entity, st, correct):
#     y_pred = [0 for z in range(len(entity))]
#     y_true = [1 for z in range(len(entity))]
#     for k in st.lstrip().rstrip().split(' '):
#         if (k in entity):
#             y_pred[entity.index(k)] = 1
#     return y_true,y_pred


class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        # https://blog.csdn.net/xizero00/article/details/51182003
        '''End to End memory networks'''
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)   # C.weight.data get the weights, then normal_ ops
            '''adds a child module to the current module.
            The module can be accessed as an attribute using the given name.
            self._modules['C_'+str(hop)]   # this is embedding C
            '''
            self.add_module("C_{}".format(hop), C)

        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        
    def get_state(self,bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))

    def forward(self, story):
        story = story.transpose(0, 1)
        story_size = story.size()   # b * m * 3
        if self.unk_mask:   # mask input as unk;
            if(self.training):
                ones = np.ones((story_size[0],story_size[1],story_size[2]))
                # dropout for UNK
                '''n trials and p probability of success'''
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))],1-self.dropout)[0]
                ones[:,:,0] = ones[:,:,0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA: a = a.cuda()
                story = story*a.long()
        u = [self.get_state(story.size(0))]     # shape(B,E)
        for hop in range(self.max_hops):
            # https://stackoverflow.com/questions/48915810/pytorch-contiguous
            '''
            Where contiguous here means contiguous in memory.
            So the contiguous function doesn't affect your target tensor at all,
            it just makes sure that it is stored in a contiguous chunk of memory.
            '''
            # - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
            # - Output: `(N, W, embedding_dim)`
            # note: Embedding input is two dims;
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long())    # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),))  # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e

            # u is a list of state; state shape [B,E]; u is q vector in the paper
            # expand_as(other)  is same as expand(other.size())
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob   = self.softmax(torch.sum(m_A*u_temp, 2))         # (b,m)
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            m_C = torch.sum(embed_C, 2).squeeze(2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)       # (B,E)
            u_k = u[-1] + o_k
            u.append(u_k)   
        return u_k

class DecoderrMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(DecoderrMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        # an instance of class; init the weights and bias
        # the forward function is F.linear(input, self.weight, self.bias)
        self.W = nn.Linear(embedding_dim,1)
        self.W1 = nn.Linear(2*embedding_dim,self.num_vocab)

        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)

    def load_memory(self, story):
        story_size = story.size()   # b * m * 3
        if self.unk_mask:
            if(self.training):
                ones = np.ones((story_size[0],story_size[1],story_size[2]))
                # zero stand for UNK
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))],1-self.dropout)[0]
                ones[:,:,0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA:
                    a = a.cuda()
                story = story*a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            # m_A 即 embed_A 依赖 story和Embedding
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))   # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size+(embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            m_A = embed_A
            # TODO: 计算的浪费 ？？ 应该最后计算
            # 同理, m_C 即 embed_C 依赖 story和Embedding
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        # hop + 1 个elem; elem shape (B,M,E)
        self.m_story.append(m_C)        # 只保留最后的 m_C

    def ptrMemDecoder(self, enc_query, last_hidden):
        '''
        :param enc_query: this is current input x_t (or previously generated word) in batch form
        :param last_hidden: this is Eq.(4) h_{t-1} in the paper; 3 dims as required in the GRU docs
        :return:
        '''
        embed_q = self.C[0](enc_query)  # b * e
        '''
        Inputs: input, h_0
            **input** (seq_len, batch, input_size)
            **h_0** (num_layers * num_directions, batch, hidden_size)
        Outputs: output, h_n
            **output** (seq_len, batch, hidden_size * num_directions)
            **h_n** (num_layers * num_directions, batch, hidden_size)
        '''
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        # u is a list; elem is tensor(B,H) if B not equal 1; else elem shape(H,)
        u = [hidden[0].squeeze()]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]     # (B,M,E)
            if(len(list(u[-1].size()))==1):     # if the dim of u[-1] is 1
                u[-1] = u[-1].unsqueeze(0)      # used for bsz = 1. actually there is no the batch dim
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)      # shape (B,1,E) ---> (B,M,E)
            prob_lg = torch.sum(m_A*u_temp, 2)          # (B,M)
            prob_   = self.softmax(prob_lg)             # (B,M)     attention
            m_C = self.m_story[hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)    # (B,M,E)
            o_k  = torch.sum(m_C*prob, 1)               # (B,E)
            if (hop==0):
                p_vocab = self.W1(torch.cat((u[0], o_k),1))  # Eq.(5) in paper; shape(B,V)
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg         # the last, actually is logits  shape(B,M)
        return p_ptr, p_vocab, hidden


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module    # an instance of NN_model
        self.prefix = prefix

    # to use list
    def __getitem__(self, i):
        # getattr(object, name[, default]) -> value
        return getattr(self.module, self.prefix + str(i))

if __name__ == '__main__':
    from ../utils.utils_babi_mem2seq import *
    lang, max_len, max_r = prepare_data_seq(args['task'], batch_size=int(args['batch']))
    model = Mem2Seq(nt(args['hidden']),max_len,max_r,lang,args['path'],args['task'], lr=0.0, n_layers=int(args['layer']), dropout=0.0, unk_mask=0)

    while True:
        user_input = input()
        response = model.demo_res(user_input)