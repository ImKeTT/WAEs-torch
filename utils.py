#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: utils.py
@author: ImKe at 2021/8/13
@email: thq415_ic@yeah.net
@feature: #Enter features here
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
import random
import sys
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class TiedEmbeddingsTransposed(nn.Module):
   """Layer for tying embeddings in an output layer.
   A regular embedding layer has the shape: V x H (V: size of the vocabulary. H: size of the projected space).
   In this layer, we'll go: H x V.
   With the same weights than the regular embedding.
   In addition, it may have an activation.
   # References
       - [ Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
   """

   def __init__(self, tied_to=None, activation=None):
       super().__init__()
       self.tied_to = tied_to
       self.activation = activation
       self.transposed_weights = torch.transpose(self.tied_to.weight, -2, -1)


   def compute_output_shape(self, input_shape):
       return input_shape[0], input_shape[1], self.tied_to.weight[0].size()

   def forward(self, inputs, mask=None):
       output = torch.matmul(inputs, self.transposed_weights)
       if self.activation is not None:
           output = self.activation(output)
       return output

class PositionalEmbedding(nn.Module):
    def __init__(self, size=None, mode='sum', dropout=0.1):
        """
        positional embedding class, batch_first=False to fit customized TransformerDecoder
        :param size:
        :param mode:
        :param dropout:
        """
        super().__init__()
        self.size = size
        self.mode = mode
        self.dropout = nn.Dropout(p=dropout)
        self.device = "cuda"

   # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, x):
       if (self.size == None) or (self.mode == 'sum'):
           self.size = int(x.shape[-1])
       batch_size, seq_len = x.size(0), x.size(1)
       position_j = 1. / torch.pow(10000., 2 * torch.arange(0, self.size / 2) / self.size)
       position_j = position_j.unsqueeze(0).to(self.device)
       position_i = torch.cumsum(torch.ones_like(x[:, :, 0]), 1) - 1
       position_i = position_i.unsqueeze(2).to(self.device)
       position_ij = torch.matmul(position_i, position_j) # cannot use torch.mm(), which requires the same dim of both input, but K.dot works
       position_ij = torch.cat([torch.cos(position_ij), torch.sin(position_ij)], 2)
       if self.mode == 'sum':
           return self.dropout(position_ij + x)
       elif self.mode == 'concat':
           return self.dropout(torch.cat([position_ij, x], 2))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # ninp, dropout
        """
        positional embedding class, batch_first=False to fit torch.nn.TransformerDecoderLayer()
        :param d_model:
        :param dropout:
        :param max_len:
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 5000 * 200
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [[0],[1],...[4999]] 5000 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(
            10000.0) / d_model))  # e ^([0, 2,...,198] * -ln(10000)(-9.210340371976184) / 200) [1,0.912,...,(1.0965e-04)]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len=5000, 1, emb] #5000 * 1 * 200, 最长5000的序列，每个词由1 * 200的矩阵代表着不同的时间
        self.register_buffer('pe', pe)

    def forward(self, x):
        ## x [max_len, bs, emb_size]
        x = x + self.pe[x.size(0), :]        # torch.Size([35, 1, 200])
        return self.dropout(x)

class Attention(nn.Module):
    # uni-directional self attention for long-range text generation
    def __init__(self, nb_head, size_per_head, max_len, att_size, device="cuda"):
        super().__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.max_len = max_len
        self.WQ = nn.Parameter(torch.randn(att_size[0][-1], self.output_dim))
        self.WK = nn.Parameter(torch.randn(att_size[1][-1], self.output_dim))
        self.WV = nn.Parameter(torch.randn(att_size[2][-1], self.output_dim))
        self.device = device

    # def build(self, input_shape):
    #     self.WQ = self.add_weight(name='WQ',
    #                               shape=(input_shape[0][-1], self.output_dim),
    #                               initializer='glorot_uniform',
    #                               trainable=True)
    #     self.WK = self.add_weight(name='WK',
    #                               shape=(input_shape[1][-1], self.output_dim),
    #                               initializer='glorot_uniform',
    #                               trainable=True)
    #     self.WV = self.add_weight(name='WV',
    #                               shape=(input_shape[2][-1], self.output_dim),
    #                               initializer='glorot_uniform',
    #                               trainable=True)
    #     super(Attention, self).build(input_shape)

    def Mask(self, inputs):
        mask = torch.eye(self.max_len)  # [ml, ml]
        mask = torch.cumsum(mask, 1)  # [ml,ml]
        mask = mask.unsqueeze(0)  # [bs, ml, ml]

        eye = torch.eye(self.max_len)
        eye = eye.unsqueeze(0)
        mask = mask - eye

        mask = mask.unsqueeze(1)  # [1,1, ml,ml]
        mask = mask.permute(0, 3, 2, 1).to(self.device)

        return inputs - mask * 1e12

    def forward(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x

        Q_seq = torch.matmul(Q_seq , self.WQ)  # [bs, ml, output_dim]
        Q_seq = torch.reshape(Q_seq, (-1, Q_seq.size(1), self.nb_head, self.size_per_head))  # [bs, ml, nb_head, size_per_head]
        Q_seq = Q_seq.permute(0, 2, 1, 3)  # [bs, nb_head, ml, size_per_head]
        K_seq = torch.matmul(K_seq , self.WK)  # [bs, ml, output_dim]
        K_seq = torch.reshape(K_seq, (-1, K_seq.size(1), self.nb_head, self.size_per_head))
        K_seq = K_seq.permute(0, 2, 1, 3)  # [bs, nb_head, ml, size_per_head]
        V_seq = torch.matmul(V_seq , self.WV)
        V_seq = torch.reshape(V_seq, (-1, V_seq.size(1), self.nb_head, self.size_per_head))
        V_seq = V_seq.permute(0, 2, 1, 3)  # [bs, nb_head, ml, size_per_head]

        # attention score
        A = torch.matmul(Q_seq, torch.transpose(K_seq, 2, 3)) / self.size_per_head ** 0.5  # [bs, nb_head, ml, ml]
        # print(A.size())
        A = A.permute(0, 3, 2, 1)  # [bs, ml, ml, nb_head]
        A = self.Mask(A)
        A = A.permute(0, 3, 2, 1)  # [bs, nb_head, ml, ml]
        # turn to probability distribution
        A = nn.Softmax(dim=-1).to(self.device)(A)

        O_seq = torch.matmul(A, V_seq)  # [bs, nb_head, ml, size_per_head]
        O_seq = O_seq.permute(0, 2, 1, 3)  # [bs, ml, nb_head, size_per_head]
        # O_seq not continuous, cannot use .view()
        O_seq = torch.reshape(O_seq, (-1, O_seq.size(1), self.output_dim)) # [bs, ml, nb_head * size_per_head]
        return O_seq


class LayerNormalization(nn.Module):
    """
        Implementation according to:
            "Layer Normalization" by JL Ba, JR Kiros, GE Hinton (2016)
    """

    def __init__(self, x_size, epsilon=1e-8):
        super().__init__()
        self._epsilon = epsilon
        self._g = nn.Parameter(torch.ones(x_size,))
        self._b = nn.Parameter(torch.zeros(x_size,))

    def forward(self, x):
        mean = torch.mean(x, axis=-1)
        std = torch.std(x, axis=-1)

        if len(x.size()) == 3:
            mean = mean.unsqueeze(1).repeat(1, x.size(-1), 1).permute(0, 2, 1)
            std = std.unsqueeze(1).repeat(1, x.size(-1), 1).permute(0, 2, 1)

        elif len(x.size()) == 2:
            mean = torch.reshape(mean.repeat(x.size(-1), 1),
                (-1, x.size(-1))
            )
            std = torch.reshape(mean.repeat(x.size(-1), 1),
                (-1, x.size(-1))
            )
        # print(self._b.size(), self._g.size(), mean.size())
        return self._g * (x - mean) / (std + self._epsilon) + self._b

def sample(preds, diversity=1.0):
    # sample from te given prediction
    # preds = np.asarray(preds).astype('float64')
    preds = torch.log(preds) / diversity
    exp_preds = preds.exp()
    preds = exp_preds / torch.sum(exp_preds)
    probas = torch.multinomial(preds, 1)
    return probas

def argmax(preds):
    # preds = np.asarray(preds).astype('float64')
    return torch.argmax(preds)

def padding(batch_data, pad_token, max_len=None):
    cur_maxlen = max([len(i) for i in batch_data])
    _batch_data = [ii[:] for ii in batch_data]
    max_len = max_len if max_len != None else cur_maxlen
    for i in range(len(_batch_data)):
        if len(_batch_data[i]) < max_len:
            _batch_data[i] += [pad_token] * (max_len - len(_batch_data[i]))
        else:
            _batch_data[i] = _batch_data[i][:max_len]
    return np.array(_batch_data)

################
# MMD Distance #
################
## gaussian kernel
def compute_kernel(x, y, kernel, imq_c):
    """
    gaussian kernel
    :param x: point x
    :param y: point y
    :return: kernel distance between x, y
    """
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
    euclidean_dist = torch.mean((tiled_x - tiled_y) ** 2, dim=2)
    if kernel == "RBF":
        computed_kernel = torch.exp(-euclidean_dist / dim * 1.0)
    elif kernel == "IMQ":
        computed_kernel = 1. / (euclidean_dist + imq_c)
    return computed_kernel
def mmd(x, y, kernel="RBF", imq_c=1):
    """
    mmd distance
    :param x: point x
    :param y: point y
    :return: mmd distance between x, y
    """
    x_kernel = compute_kernel(x, x, kernel, imq_c)
    y_kernel = compute_kernel(y, y, kernel, imq_c)
    xy_kernel = compute_kernel(x, y, kernel, imq_c)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

"""
def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor):
    h_dim = X.size(-1)
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    ## scale is the hyperparam to be tuned, here use averaged method
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

## Gaussian Kernel
def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor):
    h_dim = X.size(-1)
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats
"""

## free or forze parameters for GAN training
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def kl_anneal_function(anneal_function, step, total_step, cyc_num=10, propotion=0.3):
    step = step%(int(total_step/cyc_num))
    ## cycli- cal schedule to anneal β for 10 periods, with training AE (β = 0) for 0.5 proportion
    if step<=int((total_step/cyc_num)*propotion):
        if anneal_function is None:
            return 1.0
        else:
            return 0.0
    else:
        step -= int((total_step/cyc_num)*propotion)
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-0.005 * (step - 5000))))
        elif anneal_function == 'linear':
            return min(1, step / int((total_step/cyc_num)*((1-propotion)/2)))
        elif anneal_function == 'tanh':
            return ((np.tanh((step - 5000) / 1000) + 1) / 2).item()
        elif anneal_function == "const" or anneal_function is None:
            return 1.0

class Dataset(nn.Module):
    def __init__(self, dataname, max_len, max_vocab, batch_size, logger, device="cuda"):
        super().__init__()
        self.pad_token = 0
        self.unk_token = 1
        self.sos_token = 2
        self.eos_token = 3
        self.data = dataname
        self.max_len = max_len
        self.bs = batch_size
        self.max_vocab = max_vocab
        self.logger = logger
        self.device=device

    def read_data(self):
        # prefix_root = "/data1/tuhq/rnn-stega-torch/text-vae/topic-vae/topic"
        prefix_root = "/data/tuhq/toemlm/WAE"
        data_name = self.data
        max_vocab = self.max_vocab
        root = f"{prefix_root}/data/{data_name}"
        train = []
        val = []
        test = []
        with open(f"{root}/train.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                train.append(line)
        self.logger.info(f"train corpus size: {sum([len(d) for d in train])}")
        sys.stdout.flush()
        self.logger.info(f"sequences: {len(train)}")
        sys.stdout.flush()

        vocab_path = f"{root}/{data_name}-vocab.json"
        if os.path.exists(vocab_path):
            chars, id2char, char2id = json.load(open(vocab_path))
            id2char = {int(i): j for i, j in id2char.items()}
        else:
            chars = {}
            for lyric in train:
                for w in lyric:
                    chars[w] = chars.get(w, 0) + 1

            self.logger.info(f"all vocab: {len(chars)}")
            sys.stdout.flush()

            sort_chars = sorted(chars.items(), key=lambda a: a[1], reverse=True)
            self.logger.info(sort_chars[:10])
            sys.stdout.flush()
            chars = dict(sort_chars[:max_vocab])

            id2char = {i + 4: j for i, j in enumerate(chars)}

            id2char[self.sos_token] = "<SOS>"
            id2char[self.eos_token] = "<EOS>"
            id2char[self.unk_token] = "<UNK>"
            id2char[self.pad_token] = "<PAD>"

            char2id = {j: i for i, j in id2char.items()}
            json.dump([chars, id2char, char2id], open(vocab_path, "w"))

        self.char2id = char2id
        self.id2char = id2char
        self.vocab_size = len(char2id)
        self.logger.info(f"vocab size: {self.vocab_size}")
        sys.stdout.flush()

        with open(f"{root}/valid.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                val.append(line)

        with open(f"{root}/test.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower().split(' ')
                test.append(line)
        self.train = train
        self.test = test
        self.val = val

    def str2id(self, s, start_end = False):
        ids = [self.char2id.get(c, self.unk_token) for c in s]
        if start_end:
            ids = [self.sos_token] + ids + [self.eos_token]
        return ids

    def id2str(self, ids):
        return [self.id2char[x] for x in ids]

    def padding(self, x, y, z):
        ml = self.max_len
        x = [i + [0] * (ml - len(i)) for i in x]
        y = [i + [0] * (ml - len(i)) for i in y]
        z = [i + [0] * (ml - len(i)) for i in z]
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        return x, y, z

    def data_generator(self, data):
        x = []
        y = []
        z = []

        while True:
            np.random.shuffle(data)
            for d in data:
                if len(d) > (self.max_len - 2):
                    d = d[:self.max_len - 2]

                d = self.str2id(d, start_end=True)

                x.append(d)
                y.append(d)
                z.append(d[1:])

                if len(x) == self.bs:
                    x, y, z = self.padding(x, y, z)

                    yield torch.tensor([x, y, z]), None
                    x = []
                    y = []
                    z = []

    def get_batch_num(self, data, max_len, batch_size):
        x = []
        y = []
        z = []
        bs_num = 0

        for d in data:
            if len(d) > (max_len - 2):
                d = d[:max_len - 2]

            d = self.str2id(d, start_end=True)

            x.append(d)
            y.append(d)
            z.append(d[1:])

            if len(x) == batch_size:
                x, y, z = self.padding(x, y, z)

                bs_num += 1
                x = []
                y = []
                z = []

        return bs_num


    def gen_bs(self, vec, latent_dim, dec_model, topk=3):
        """beam search
        """
        self.logger.info("\nbeam search...")
        sys.stdout.flush()
        xid = [[self.sos_token]] * topk
        vec = np.reshape(np.array([vec] * topk), (topk, latent_dim))
        scores = [0] * topk
        for i in range(self.max_len):
            x_seq = torch.from_numpy(padding(xid, self.pad_token, max_len=self.max_len)).to(self.device)
            vec = torch.tensor(vec, dtype=torch.float32).to(self.device)
            proba = nn.Softmax(dim=-1)(dec_model(x_seq, vec, self.max_len))
            proba = proba[:, i, 3:]
            log_proba = torch.log(proba + 1e-6)
            arg_topk = log_proba.argsort(axis=1)[:, -topk:]
            _xid = []
            _scores = []
            if i == 0:
                for j in range(topk):
                    _xid.append(list(xid[j]) + [arg_topk[0][j].item() + 3])
                    _scores.append(scores[j] + log_proba[0][arg_topk[0][j].item()])
            else:
                for j in range(len(xid)):
                    for k in range(topk):
                        _xid.append(list(xid[j]) + [arg_topk[j][k].item() + 3])
                        _scores.append(scores[j] + log_proba[j][arg_topk[j][k].item()])
                _arg_topk = np.argsort(_scores)[-topk:]
                _xid = [_xid[k] for k in _arg_topk]
                _scores = [_scores[k] for k in _arg_topk]
            yid = []
            scores = []
            for k in range(len(xid)):
                yid.append(_xid[k])
                scores.append(_scores[k])
            xid = yid

        s = self.id2str(xid[np.argmax(scores)])
        self.logger.info(' '.join(s))
        sys.stdout.flush()

    def gen_from_vec(self, diversity, vec, argmax_flag, dec_model):
        start_index = self.sos_token  # <BOS>
        start_word = self.id2char[start_index]
        self.logger.info("-"*20)

        generated = [[start_index]]
        # sys.stdout.write(start_word)
        words = [start_word]

        while (self.eos_token not in generated[0] and len(generated[0]) <= self.max_len):
            x_seq = torch.from_numpy(padding(generated, self.pad_token, self.max_len)).to(self.device)
            preds = dec_model(x_seq, vec, self.max_len)[0]
            preds = nn.Softmax(dim=-1)(preds)
            preds = preds[len(generated[0]) - 1][3:]
            if argmax_flag:
                next_index = argmax(preds)
            else:
                next_index = sample(preds, diversity)
            next_index = next_index.item()
            next_index += 3
            next_word = self.id2char[next_index]

            generated[0] += [next_index]
            words.append(next_word)
            # self.logger.info(next_word + " ")
            # sys.stdout.write(next_word + ' ')
            # sys.stdout.flush()
        self.logger.info(' '.join(words))

    def reconstruct(self, num, test_gen, enc_model, dec_model):
        for i in range(num):
            self.logger.info('\nreconstructing, first false second true')
            s = next(test_gen)[0][0][0]
            s_w = ' '.join([self.id2char[x.item()] for x in s])
            self.logger.info(s_w)
            sys.stdout.flush()
            _, _, s_v = enc_model(torch.stack([s]).to(self.device))
            self.gen_from_vec(0.8, s_v, False, dec_model)
            self.gen_from_vec(0.8, s_v, True, dec_model)

    def interpolate(self, diversity, num, test_gen, enc_model, dec_model):
        s1 = next(test_gen)[0][0][0].unsqueeze(0).to(self.device)
        s2 = next(test_gen)[0][0][0].unsqueeze(0).to(self.device)

        _, _, vec1 = enc_model(s1)
        _, _, vec2 = enc_model(s2)
        s1 = s1.cpu().numpy().tolist()[0]
        s2 = s2.cpu().numpy().tolist()[0]
        self.logger.info('interpolate with sampling')
        self.logger.info(' '.join([self.id2char[x] for x in s1]))
        sys.stdout.flush()
        for i in range(1, num + 1):
            alpha = i / (num + 1)
            vec = (1 - alpha) * vec1 + alpha * vec2
            self.gen_from_vec(diversity, vec, False, dec_model)
        self.logger.info(' '.join([self.id2char[x] for x in s2]))
        sys.stdout.flush()

        self.logger.info('interpolate with argmax')
        self.logger.info(' '.join([self.id2char[x] for x in s1]))
        sys.stdout.flush()
        for i in range(1, num + 1):
            alpha = i / (num + 1)
            vec = (1 - alpha) * vec1 + alpha * vec2
            self.gen_from_vec(diversity, vec, True, dec_model)
        self.logger.info(' '.join([self.id2char[x] for x in s2]))
        sys.stdout.flush()

    def gen(self, argmax_flag, latent_dim, dec_model):
        random_vec = torch.randn(1, latent_dim).to(self.device)

        start_index = self.sos_token  # <BOS>
        start_word = self.id2char[start_index]

        for diversity in [0.5, 0.8, 1.0]:
            for j in range(1):
                self.logger.info(f"----- diversity: {diversity}")

                generated = [[start_index]]
                self.logger.info('----- Generating -----')
                # self.logger.info(start_word)
                out_words = []
                # sys.stdout.flush()

                while (self.eos_token not in generated[0] and len(generated[0]) <= self.max_len):
                    x_seq = torch.from_numpy(padding(generated, self.pad_token, self.max_len)).to(self.device)
                    preds = dec_model(x_seq, random_vec, self.max_len)[0]
                    preds = nn.Softmax(dim=-1)(preds)
                    preds = preds[len(generated[0]) - 1][3:] # exclude special tokens except eos_token
                    if argmax_flag:
                        next_index = argmax(preds).item()
                    else:
                        next_index = sample(preds, diversity).item()

                    next_index += 3 ## add special token number
                    next_word = self.id2char[next_index]

                    generated[0] += [next_index]
                    out_words.append(next_word)
                    # self.logger.info(next_word + ' ')
                    # sys.stdout.flush()
                self.logger.info(' '.join(out_words))
                # self.logger.info(" ")
