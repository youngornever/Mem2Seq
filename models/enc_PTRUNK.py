import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
import datetime
from utils.measures import wer,moses_multi_bleu
from tqdm import tqdm
from sklearn.metrics import f1_score
import math

class PTRUNK(nn.Module):
    def __init__(self,hidden_size,max_len,max_r,lang,path,task,lr,n_layers, dropout):
        super(PTRUNK, self).__init__()
        self.name = "PTRUNK"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len  # max input
        self.max_r = max_r  # max response len
        self.lang = lang
        self.lr = lr
        self.decoder_learning_ratio = 5.0
        self.n_layers = n_layers
        self.dropout = dropout
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
                self.decoder.viz_arr =[] 
        else:
            self.encoder = EncoderRNN(lang.n_words, hidden_size, n_layers,dropout)
            self.decoder = PtrDecoderRNN(hidden_size, lang.n_words, n_layers, dropout)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr* self.decoder_learning_ratio)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_gate = 0
        self.loss_ptr = 0
        self.loss_vac = 0       # vocab; 作者拼错了
        self.print_every = 1
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, VL:{:.2f},GL:{:.2f}, PL:{:.2f}'.format(print_loss_avg,print_loss_vac,print_loss_gate,print_loss_ptr)
    
    def save_model(self,dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        if USEKB:
            directory = 'save/PTR_KB-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)         
        else:
            directory = 'save/PTR_noKB-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)         
        #directory = 'save/PTR_KVR_KB/'+str(self.task)+'HDD'+str(self.hidden_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type) #+datetime.datetime.now().strftime("%I%M%p%B%d%Y"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')
        
    def train_batch(self, input_batches, input_lengths, target_batches, 
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio,reset):
        '''
        :param input_batches:       (T,B)
        :param input_lengths:       (B,)
        :param target_batches:      (T,B)
        :param target_lengths:      (B,)
        :param target_index:        (T,B)
        :param target_gate:         (T,B)
        :param batch_size:          B
        :param clip:
        :param teacher_forcing_ratio:
        :param reset:
        :return:
        '''
        if reset:
            self.loss = 0
            self.loss_gate = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1 
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab,loss_Ptr,loss_Gate = 0,0,0

        # Run words through encoder
        # **output** (seq_len, batch, hidden_size * num_directions)
        # **h_n** (num_layers * num_directions, batch, hidden_size)
        # **c_n** (num_layers * num_directions, batch, hidden_size)
        # encoder_hidden = (H_n, c_n)
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths)
      
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        # TODO: encoder可能双向？？ decoder只能单项？？
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],encoder_hidden[1][:self.decoder.n_layers])
        
        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, batch_size, encoder_outputs.size(0)))
        all_decoder_outputs_gate = Variable(torch.zeros(max_target_length, batch_size))
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:    
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr,decoder_vacab,gate,decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                all_decoder_outputs_gate[t] = gate
                decoder_input = target_batches[t]   # Next input is current target
                if USE_CUDA: decoder_input = decoder_input.cuda()
                
        else:
            for t in range(max_target_length):
                # (B*1*T) (1,B.Out_size)  (B,1)   a tuple of two elem; (num_layers, batch, hidden_size)

                decoder_ptr, decoder_vacab, gate, decoder_hidden = self.decoder(
                                        decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                all_decoder_outputs_gate[t] = gate
                topv, topvi = decoder_vacab.data.topk(1)    # (B,1)
                topp, toppi = decoder_ptr.data.topk(1)
                # get the correspective word in input
                top_ptr_i = torch.gather(input_batches,0,Variable(toppi.view(1, -1)))
                next_in = [top_ptr_i.squeeze()[i].data[0] if(gate.squeeze()[i].data[0]>=0.5) else topvi.squeeze()[i] for i in range(batch_size)]
                decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
                  
        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), # -> batch x seq
            target_index.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )
        # FIXME: 这种形式的loss好像不太好;gate is \in (0,1)
        loss_gate = self.criterion(all_decoder_outputs_gate,target_gate.float())


        loss = loss_Vocab + loss_Ptr + loss_gate
        loss.backward()
        
        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.data[0]
        self.loss_gate += loss_gate.data[0] 
        self.loss_ptr += loss_Ptr.data[0]
        self.loss_vac += loss_Vocab.data[0]
        
        
    def evaluate_batch(self,batch_size,input_batches, input_lengths, target_batches, target_lengths, target_index,target_gate,src_plain):  
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],encoder_hidden[1][:self.decoder.n_layers])

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.decoder.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, encoder_outputs.size(0)))
        all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()
        p = []
        for elm in src_plain:
            p.append(elm.split(' '))
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr, decoder_vacab, gate, decoder_hidden = self.decoder(
                                    decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs_vocab[t] = decoder_vacab
            all_decoder_outputs_ptr[t] = decoder_ptr
            all_decoder_outputs_gate[t] = gate

            topv, topvi = decoder_vacab.data.topk(1)
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches,0,Variable(toppi.view(1, -1)))
            next_in = [top_ptr_i.squeeze()[i].data[0] if(gate.squeeze()[i].data[0]>=0.5) else topvi.squeeze()[i] for i in range(batch_size)]
            decoder_input = Variable(torch.LongTensor(next_in)) 
            # Next input is chosen word
            if USE_CUDA: decoder_input = decoder_input.cuda()

            temp = []
            for i in range(batch_size):
                if(gate.squeeze()[i].data[0]>=0.5):     # generate
                    if(toppi.squeeze()[i] >= len(p[i])):    # 当生成的时候， 如果ptr大于sentence_end 则结束
                        temp.append('<EOS>')
                    else:
                        temp.append(p[i][toppi.squeeze()[i]])   # 选取
                else:
                    ind = topvi.squeeze()[i]
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind])
            decoded_words.append(temp)      # (T,B)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words

    def evaluate(self,dev,avg_best,BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        acc_G = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED,microF1_PRED_cal,microF1_PRED_nav,microF1_PRED_wet = [],[],[],[]
        microF1_TRUE,microF1_TRUE_cal,microF1_TRUE_nav,microF1_TRUE_wet = [],[],[],[]
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar:
            # (T,B) a list of list
            words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],data_dev[2],data_dev[3],data_dev[4],data_dev[5],data_dev[6])            
            acc=0
            w = 0
            temp_gen = []
            for i, row in enumerate(np.transpose(words)):       # (B,T)
                st = ''
                for e in row:
                    if e== '<EOS>':
                        break
                    else:
                        st+= e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]
                ### compute F1 SCORE  
                if(len(data_dev)>10):
                    f1_true,f1_pred = computeF1(data_dev[8][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
                    microF1_TRUE += f1_true
                    microF1_PRED += f1_pred

                    f1_true,f1_pred = computeF1(data_dev[9][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += f1_pred 

                    f1_true,f1_pred = computeF1(data_dev[10][i],st.lstrip().rstrip(),correct.lstrip().rstrip())
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += f1_pred 

                    f1_true,f1_pred = computeF1(data_dev[11][i],st.lstrip().rstrip(),correct.lstrip().rstrip()) 
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += f1_pred  
                
                if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                    acc+=1
                w += wer(correct.lstrip().rstrip(),st.lstrip().rstrip())
                ref.append(str(correct.lstrip().rstrip()))
                hyp.append(str(st.lstrip().rstrip()))
                ref_s+=str(correct.lstrip().rstrip())+ "\n"
                hyp_s+=str(st.lstrip().rstrip()) + "\n"

            acc_avg += acc/float(len(data_dev[1]))
            wer_avg += w/float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),wer_avg/float(len(dev))))
        if(len(data_dev)>10):
            logging.info("F1 SCORE:\t"+str(f1_score(microF1_TRUE, microF1_PRED, average='micro')))
            logging.info("F1 CAL:\t"+str(f1_score(microF1_TRUE_cal, microF1_PRED_cal, average='micro')))
            logging.info("F1 WET:\t"+str(f1_score(microF1_TRUE_wet, microF1_PRED_wet, average='micro')))
            logging.info("F1 NAV:\t"+str(f1_score(microF1_TRUE_nav, microF1_PRED_nav, average='micro')))

        if (BLEU):       
            bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True) 
            logging.info("BLEU SCORE:"+str(bleu_score))     
                                                                      
            if (bleu_score >= avg_best):
                self.save_model(str(self.name)+str(bleu_score))
                logging.info("MODEL SAVED")
            return bleu_score
        else:              
            acc_avg = acc_avg/float(len(dev))
            if (acc_avg >= avg_best):
                self.save_model(str(self.name)+str(acc_avg))
                logging.info("MODEL SAVED")
            return acc_avg

def computeF1(entity,st,correct):
    y_pred = [0 for z in range(len(entity))]
    y_true = [1 for z in range(len(entity))]
    for k in st.lstrip().rstrip().split(' '):
        if (k in entity):
            y_pred[entity.index(k)] = 1
    return y_true,y_pred

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout       
        self.embedding = nn.Embedding(input_size, hidden_size)  # an instance
        self.embedding_dropout = nn.Dropout(dropout)            # an instance
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout)
        if USE_CUDA:
            self.lstm = self.lstm.cuda() 
            self.embedding_dropout = self.embedding_dropout.cuda()
            self.embedding = self.embedding.cuda() 

    def get_state(self, input):
        """Get cell states and hidden states."""
        # input shape (T,B)
        batch_size = input.size(1)      # B
        c0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))  
        # self.n_layers * self.num_directions = 2 if bi
        h0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if USE_CUDA:
            h0_encoder = h0_encoder.cuda()
            c0_encoder = c0_encoder.cuda() 
        return (h0_encoder, c0_encoder)

    def forward(self, input_seqs, input_lengths, hidden=None):

        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)       # (T,B,E)
        embedded = self.embedding_dropout(embedded)
        hidden = self.get_state(input_seqs)
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        '''
        Inputs: input, (h_0, c_0)
            - **input** (seq_len, batch, input_size): tensor containing the features
              of the input sequence.
              The input can also be a packed variable length sequence.
              See :func:`torch.nn.utils.rnn.pack_padded_sequence` for details.
            - **h_0** (num_layers \* num_directions, batch, hidden_size): tensor
              containing the initial hidden state for each element in the batch.
            - **c_0** (num_layers \* num_directions, batch, hidden_size): tensor
              containing the initial cell state for each element in the batch.

              If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.

        Outputs: output, (h_n, c_n)
            - **output** (seq_len, batch, hidden_size * num_directions): tensor
              containing the output features `(h_t)` from the last layer of the RNN,
              for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
              given as the input, the output will also be a packed sequence.
            - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
              containing the hidden state for t=seq_len
            - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
              containing the cell state for t=seq_len
        '''
        outputs, hidden = self.lstm(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        
        return outputs, hidden

class PtrDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(PtrDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size      # Vocab size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        # TODO: encoder 是 bi-LSTM ??
        self.lstm = nn.LSTM(2*hidden_size, hidden_size, n_layers, dropout=dropout)
        self.W1 = nn.Linear(2*hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)  
        self.U = nn.Linear(hidden_size, output_size)
        self.W = nn.Linear(hidden_size, 1)

        if USE_CUDA:
            self.embedding = self.embedding.cuda()
            self.embedding_dropout = self.embedding_dropout.cuda()
            self.lstm = self.lstm.cuda()
            self.W1 = self.W1.cuda() 
            self.v = self.v.cuda() 
            self.U = self.U.cuda() 
            self.W = self.W.cuda() 

    def forward(self, input_seq, last_hidden, encoder_outputs):
        '''
        :param input_seq:               (B,)
        :param last_hidden:   a tuple of two elem; (num_layers, batch, hidden_size)
        :param encoder_outputs:  (seq_len, batch, hidden_size * num_directions); num_dir = 1
        :return:
        '''
        # Note: we run this one step at a time     
        # Get the embedding of the current input word (last output word)
        max_len = encoder_outputs.size(0)
        batch_size = input_seq.size(0)
        input_seq = input_seq
        encoder_outputs = encoder_outputs.transpose(0,1)    # shape (B,max_len, H*num_dir)
            
        word_embedded = self.embedding(input_seq)  # S=1 x B x N; 此处还没有1; need to unsqueeze()
        word_embedded = self.embedding_dropout(word_embedded)

        # ATTENTION CALCULATION                     last_hidden (H_n, c_n)
        s_t = last_hidden[0][-1].unsqueeze(0)       # shape (1,B,H)
        H = s_t.repeat(max_len,1,1).transpose(0,1)  # shape (B,max_len, H)

        energy = F.tanh(self.W1(torch.cat([H,encoder_outputs], 2)))     # (B,max_len,H)
        energy = energy.transpose(2,1)                                  # (B,H,max_len)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) # [B*1*H]
        p_ptr = torch.bmm(v,energy)   # [B*1*T]
        
        a = F.softmax(p_ptr)    # dim = len(p_ptr.data.size())-1
        context = a.bmm(encoder_outputs)    # [B*1*T] * [B,T,H] ---> [B,1,H]

        # Combine embedded input word and attended context, run through RNN
        # (1,B,2*H)
        # TODO: for the case of B = 1
        rnn_input = torch.cat((word_embedded, context.squeeze(1)), 1).unsqueeze(0)
        '''
            Inputs: input, (h_0, c_0)
                - **input** (seq_len, batch, input_size): tensor containing the features
                  of the input sequence.
                  The input can also be a packed variable length sequence.
                  See :func:`torch.nn.utils.rnn.pack_padded_sequence` for details.
                - **h_0** (num_layers \* num_directions, batch, hidden_size): tensor
                  containing the initial hidden state for each element in the batch.
                - **c_0** (num_layers \* num_directions, batch, hidden_size): tensor
                  containing the initial cell state for each element in the batch.

                  If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.

            Outputs: output, (h_n, c_n)
                - **output** (seq_len, batch, hidden_size * num_directions): tensor
                  containing the output features `(h_t)` from the last layer of the RNN,
                  for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                  given as the input, the output will also be a packed sequence.
                - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the hidden state for t=seq_len
                - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the cell state for t=seq_len
        '''
        # output shape (1,B,H);  为什么squeeze都不写？？
        output, hidden = self.lstm(rnn_input, last_hidden)
        p_vacab = self.U(output)        # (1,B,Out_size)  ???
        
        gate = F.sigmoid(self.W(hidden[0][-1]))     # (B,1)
        # # (B*1*T) (1,B.Out_size)  (B,1)   a tuple of two elem; (num_layers, batch, hidden_size)
        return p_ptr, p_vacab, gate, hidden
