import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN_ENCODER_attention(nn.Module):
    def __init__(self, ntoken, hyparameter, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER_attention, self).__init__()
        self.n_steps = hyparameter['word_num']
        self.ntoken = ntoken
        self.ninput = ninput
        self.drop_prob = drop_prob
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.rnn_type = hyparameter['rnn_type']
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

        self.style_dim = hyparameter['TEXT_EMBEDDING_DIM']
        self.linear_out = nn.Linear(self.style_dim*2, self.style_dim)

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)

        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()

        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)

        seq_len = words_emb.shape[2]
        nef = words_emb.shape[1]
        batch_size = words_emb.shape[0]

        sent_emb_copy = sent_emb.clone()
        words_emb_copy = words_emb.clone()
        sent_emb_copy = sent_emb_copy.unsqueeze(-1)
        words_emb_copy = words_emb_copy.transpose(1, 2)
        attn = torch.bmm(words_emb_copy, sent_emb_copy)
        attn = F.softmax(attn.view(batch_size, seq_len), dim=1).view(batch_size, 1, seq_len)
        mix = torch.bmm(attn, words_emb_copy)
        mix = mix.view(batch_size, nef)

        combined = torch.cat((mix, sent_emb), dim=1)
        output_final = F.tanh(self.linear_out(combined))
        return words_emb, output_final

