#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: WAE_mmd.py
@author: ImKe at 2021/8/21
@email: thq415_ic@yeah.net
@feature: #Enter features here
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import datetime
import random

from logger import Logger
from utils import *

####################
# Hyper Parameters #
####################
max_len = 20
max_vocab = 30000
clip = 5.0 ## max gradient clip

emb_size = 256
gru_dim = 150
batch_size = 512
latent_dim = 64
hidden_dim = 128
# nambda = 20
head_num = 8
head_size = [(emb_size + latent_dim) // head_num,
             (emb_size + 2*latent_dim) // head_num,
             (emb_size + 3*latent_dim) // head_num] # for self attention
iterations = 200000 # total training iteration
epochs = 10
max_batch_iter = 15000
log_iter = 200
alpha = 50.0 # weight of VAE normalization term
p = 6 # for gradient computation
k = 2 # for gradient computation
wd = 1e-4 # weight decay of optimizer
dataname = "apnews" # data name
kernel = "RBF" # kernel method for MMD
imq_c = 1 # parameter c for IMQ kernel method (default 1)
seed = 42
lr = 0.0001
stop = 500
add_kl = True # whether add KLD as regularization
kl_anneal_func = "const" # work when add_kl is True
kl_thresh = 0.0
scratch = False
store = True # wether to store checkpoint for model training

torch.manual_seed(seed)
random.seed(seed)
now = datetime.datetime.now()
device = "cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def get_savepath(iter_n, mode):
    """
    checkpoint save path for current model
    :return:
    """
    ckpt_root = f"./ckpt/{dataname}/{mode}"
    os.makedirs(ckpt_root, exist_ok=True)
    path = f"{ckpt_root}/iter{iter_n}-emb{emb_size}.gru{gru_dim}.bs{batch_size}.latent{latent_dim}." \
           f"hiddim{hidden_dim}.alpha{alpha}.kl.{add_kl}_{kl_anneal_func}.thresh{kl_thresh}.scratch." \
           f"{scratch}.{dataname}.date{now.month}-{now.day}.pt"
    return path


##########
# Models #
##########
class Encoder(nn.Module):
    """
    Sequence Encoder
    """
    def __init__(self, emb_size, gru_dim, latent_dim, vocab_size, bidrectional=True):
        super().__init__()
        self.bidrection = bidrectional
        self.encoder = nn.GRU(emb_size, gru_dim, batch_first=True, bidirectional=bidrectional)
        if bidrectional:
            self.fclv = nn.Linear(gru_dim * 2, latent_dim)
            self.fcmu = nn.Linear(gru_dim * 2, latent_dim)
        else:
            self.fclv = nn.Linear(gru_dim, latent_dim)
            self.fcmu = nn.Linear(gru_dim, latent_dim)
        self.emb_layer = nn.Embedding(vocab_size, emb_size)

    def reparameterize(self, mean, logvar):
        sd = torch.exp(0.5 * logvar)  # Standard deviation
        # We'll assume the posterior is a multivariate Gaussian
        eps = torch.randn_like(sd)
        z = eps.mul(sd).add(mean)
        return z

    def forward(self, encoder_input):
        encoder_emb = self.emb_layer(encoder_input)
        output, hn = self.encoder(encoder_emb)
        if self.bidrection:
            hn = torch.cat([hn[0], hn[1]], dim=-1)
        lv = self.fclv(hn)
        mu = self.fcmu(hn)
        sample = self.reparameterize(mu, lv)
        return mu, lv, sample

class Decoder_torch(nn.Module):
    """
        Sequence Decoder (Transformer from torch.nn)
        """

    def __init__(self, latent_dim, embed_layer, head_n, head_size, vocab_size, dropout=0.1):
        super().__init__()
        self.embed = embed_layer ## same word embedding layer as encoder
        self.position_embed = PositionalEncoding(emb_size)
        self.head_n = head_n
        self.head_size = head_size
        self.z2src = nn.Linear(latent_dim, emb_size)
        # self.layer_norm = LayerNormalization(latent_dim * 3)
        self.latent_dim = latent_dim
        ## torch v1.4.0 does not support batch_first
        decoder_layers = nn.TransformerDecoderLayer(emb_size, head_n, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=3)
        self.final_linear = nn.Linear(emb_size, vocab_size)

    def forward(self, dec_in, enc_z, max_len=max_len):
        src = self.embed(dec_in) ## [bs, max_len, emb_size]
        src = src.transpose(0, 1) ## [max_len, bs, emb_size]
        src = self.position_embed(src) ## [max_len, bs, emb_size]
        enc_z = self.z2src(enc_z).unsqueeze(0).repeat(max_len, 1, 1) ## [max_len, bs, emb_size]
        assert enc_z.size(-1)==src.size(-1)
        output = self.transformer_decoder(src, enc_z)
        output = self.final_linear(output) ## [max_len, bs, vocab_size]
        return output.transpose(0, 1)

class Decoder(nn.Module):
    """
    Sequence Decoder (Transformer from scratch)
    """
    def __init__(self, latent_dim, embed_layer, head_n, head_size):
        super().__init__()
        # self.activation = nn.Softmax()
        self.activation = None # nn.CrossEntropy() without softmax = K.categorical_crossentropy()
        self.dec_softmax =TiedEmbeddingsTransposed(embed_layer, self.activation)
        self.embed = embed_layer
        self.position_embed = PositionalEmbedding()
        self.decoder_net = nn.ModuleList()
        self.head_n = head_n
        self.head_size = head_size
        # self.layer_norm = LayerNormalization(latent_dim * 3)
        self.latent_dim = latent_dim
        self.act2 = nn.ReLU()

        self.dense_layer = nn.ModuleList()
        self.att_layer = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.linear_layer = nn.ModuleList()
        self.final_linear = nn.Linear(emb_size + latent_dim*3, emb_size)
        for i in range(3):
            self.dense_layer.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.att_layer.append(Attention(self.head_n, self.head_size[i],
                                            max_len, [[batch_size, max_len, emb_size + latent_dim*(1+i)] for _ in range(3)]))
            self.layer_norm.append(LayerNormalization(emb_size + latent_dim*(1+i)))
            self.linear_layer.append(nn.Linear(emb_size + latent_dim*(1+i), self.head_size[i] * head_num))

    def forward(self, dec_in, enc_z, max_len=max_len):
        decoder_z = enc_z.unsqueeze(1).repeat(1, max_len, 1)
        decoder_embed = self.embed(dec_in)
        decoder_h = self.position_embed(decoder_embed)
        for layer in range(3):
            decoder_z_hier = self.dense_layer[layer](decoder_z)
            decoder_h = torch.cat([decoder_h, decoder_z_hier], -1)
            decoder_h_attn = self.att_layer[layer]([decoder_h, decoder_h, decoder_h])
            decoder_h = torch.add(decoder_h,decoder_h_attn)
            decoder_h = self.layer_norm[layer](decoder_h)
            decoder_h_mlp = self.act2(self.linear_layer[layer](decoder_h))
            decoder_h = torch.add(decoder_h, decoder_h_mlp)
            decoder_h = self.layer_norm[layer](decoder_h)
            decoder_h = self.position_embed(decoder_h)
        decoder_h = self.final_linear(decoder_h)
        decoder_output = self.dec_softmax(decoder_h)
        return decoder_output


class Decoder_(nn.Module):
    """
    Sequence Decoder
    """
    def __init__(self, latent_dim, embed_layer, head_n, head_size):
        super().__init__()
        # self.activation = nn.Softmax()
        self.activation = None # nn.CrossEntropy() without softmax = K.categorical_crossentropy()
        self.dec_softmax =TiedEmbeddingsTransposed(embed_layer, self.activation)
        self.embed = embed_layer
        self.position_embed = PositionalEmbedding()
        self.decoder_net = nn.ModuleList()
        self.head_n = head_n
        self.head_size = head_size
        # self.layer_norm = LayerNormalization(latent_dim * 3)
        self.latent_dim = latent_dim


    def forward(self, dec_in, enc_z, max_len):
        decoder_z = enc_z.unsqueeze(1).repeat(1, max_len, 1)
        decoder_embed = self.embed(dec_in)
        decoder_h = self.position_embed(decoder_embed)
        for layer in range(3):
            dense_layer = nn.Linear(self.latent_dim, self.latent_dim).to(device)
            decoder_z_hier = dense_layer(decoder_z)
            decoder_h = torch.cat([decoder_h, decoder_z_hier], -1)
            att_layer = Attention(self.head_n, self.head_size[layer], max_len,
                                       [decoder_h.size() for _ in range(3)]).to(device)
            decoder_h_attn = att_layer([decoder_h, decoder_h, decoder_h])
            decoder_h = decoder_h + decoder_h_attn
            layer_norm = LayerNormalization(decoder_h.size(-1)).to(device)
            decoder_h = layer_norm(decoder_h)
            decoder_h_mlp = nn.ReLU()(nn.Linear(decoder_h.size(-1),
                                              self.head_size[layer] * head_num).to(device)(decoder_h))
            decoder_h = decoder_h + decoder_h_mlp
            decoder_h = layer_norm(decoder_h)
            decoder_h = self.position_embed(decoder_h)
        decoder_h = nn.Linear(decoder_h.size(-1), decoder_embed.size(-1)).to(device)(decoder_h)
        decoder_output = self.dec_softmax(decoder_h)
        return decoder_output

######################
# Training functions #
######################
class EnDecoder_forward(nn.Module):
    def __init__(self, encoder, decoder, ignore_index):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ii = ignore_index

    def forward(self, data_iter):
        enc_in = data_iter[0]
        dec_in = data_iter[1]
        dec_true = data_iter[2]
        mu, lv, q_z = self.encoder(enc_in)
        kl_loss = torch.mean(- 0.5 * torch.sum(1 + lv - mu**2 - torch.exp(lv), -1), -1)

        dec_out = self.decoder(dec_in, q_z)
        dec_true_mask = (dec_true.unsqueeze(2) > 0).float()
        # nn.CrossEntropyLoss() has the reverse position w.r.t. label and output regard K.catgorical_crossentrophy
        xent_loss = torch.sum(F.cross_entropy(dec_out.contiguous().view(-1, dec_out.size(-1)), dec_true.view(-1),
                                              ignore_index=self.ii
                                              ) * dec_true_mask[:, :, 0] / torch.sum(dec_true_mask[:, :, 0]))
        p_z = torch.randn_like(q_z)
        d_loss = mmd(q_z, p_z, kernel, imq_c)
        # xent_loss /= batch_size
        all_loss = xent_loss + alpha * d_loss + kl_loss

        return kl_loss, d_loss, all_loss, xent_loss

def criterion(reconst, target, pad_token):
    return F.cross_entropy(reconst.view(-1, reconst.size(2)),
                           target.view(-1), ignore_index=pad_token)
def train(resume_file=None):
    log_dir = f"./log/{dataname}_mmd"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/emb{emb_size}.gru{gru_dim}.bs{batch_size}.latent{latent_dim}." \
               f"hiddim{hidden_dim}.alpha{alpha}.kl.{add_kl}_{kl_anneal_func}." \
               f"thresh{kl_thresh}.scratch{scratch}.data{dataname}.date{now.month}-{now.day}.txt"
    logger = Logger(log_file)
    start_epoch = 0
    #################
    # Data Iterator #
    #################
    data = Dataset(dataname, max_len, max_vocab, batch_size, logger)
    data.read_data()
    train_iter = data.data_generator(data.train)
    val_iter = data.data_generator(data.val)
    test_iter = data.data_generator(data.test)

    encoder = Encoder(emb_size, gru_dim, latent_dim, len(data.char2id)).to(device)
    if scratch:
        decoder = Decoder(latent_dim, encoder.emb_layer, head_num, head_size).to(device)
    else:
        decoder = Decoder_torch(latent_dim, encoder.emb_layer, head_num, head_size, data.vocab_size).to(device)

    # Optimizers
    enc_optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr)

    # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
    # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=30, gamma=0.5)

    ## resume if available
    if resume_file is not None:
        ckpt_enc = torch.load(f"./ckpt/{dataname}/enc/{resume_file}")
        start_epoch = ckpt_enc['epoch']
        encoder.load_state_dict(ckpt_enc['model'])
        ckpt_dec = torch.load(f"./ckpt/{dataname}/dec/{resume_file}")
        decoder.load_state_dict(ckpt_dec['model'])
        enc_optim.load_state_dict(ckpt_dec['optimizer'])

    for epoch in range(start_epoch, start_epoch + epochs):
        iter = 0
        for inum, text in enumerate(train_iter):
            # text = text.to(device)
            enc_in = text[0][0].to(device)
            dec_in = text[0][1].to(device)
            text_true = text[0][2].to(device)

            encoder.zero_grad()
            decoder.zero_grad()

            # ======== Train Generator ======== #

            free_params(decoder)
            free_params(encoder)


            mu, logvar, z_real = encoder(enc_in)
            x_recon = decoder(dec_in, z_real)
            z_fake = torch.randn_like(z_real)
            mmd_loss = mmd(z_real, z_fake, kernel, imq_c)

            recon_loss = criterion(x_recon, text_true, data.pad_token)
            kl_loss = torch.mean(- 0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), -1), -1)

            # recon_loss.unsqueeze(0).backward(one)
            # d_loss.unsqueeze(0).backward(mone)
            endecoder_loss = recon_loss + alpha * mmd_loss
            endecoder_loss.backward()
            # kl_loss = torch.mean(- 0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), -1), -1)

            enc_optim.step()
            dec_optim.step()

            iter += 1

            if (iter+1) % 50 == 0:
                logger.info("Train: Epoch: [%d/%d], Iter %d, Rec Loss: %.4f, MMD Loss: %.4f, KL Loss: %.4f" %
                      (epoch + 1, epochs, iter + 1, recon_loss/batch_size, mmd_loss, kl_loss))
            if inum >= max_batch_iter:
                break

            if (iter+1) % 5000 == 0:
                data.gen(False, latent_dim, decoder)
                data.reconstruct(2, test_iter, encoder, decoder)
                for _n in range(2):
                    vec = np.random.normal(size=(1, latent_dim))
                    data.gen_bs(vec, latent_dim, decoder, topk=1)
                # logger.info('diversity 0.8')
                # sys.stdout.flush()
                # data.interpolate(0.8, 8, test_iter, encoder, decoder)
                # logger.info('diversity 1.0')
                # sys.stdout.flush()
                # data.interpolate(1.0, 8, test_iter, encoder, decoder)

        # ======== Evaluate Training ======== #
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            test_data = next(test_iter)[0]
            # test_data = test_data.to(device)
            test_enc_in = test_data[0].to(device)
            test_dec_in = test_data[1].to(device)
            test_text_true = test_data[2].to(device)

            test_mu, _, _ = encoder(test_enc_in)
            x_test_recon = decoder(test_dec_in, test_mu, max_len)
            test_z_fake = torch.randn_like(test_mu)
            test_mmd_loss = mmd(test_mu, test_z_fake, kernel, imq_c)

            recon_loss = criterion(x_test_recon, test_text_true, data.pad_token)

            logger.info("Test: Epoch: [%d/%d], Rec Loss: %.4f, MMD Loss: %.4f" %
                  (epoch + 1, epochs, recon_loss, test_mmd_loss))

#################
# Training Main #
#################
def train2(resume_file=None):
    log_dir = f"./log/{dataname}_mmd"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/emb{emb_size}.gru{gru_dim}.bs{batch_size}.latent{latent_dim}." \
               f"hiddim{hidden_dim}.alpha{alpha}.kl.{add_kl}_{kl_anneal_func}." \
               f"thresh{kl_thresh}.scratch{scratch}.data{dataname}.date{now.month}-{now.day}.txt"
    logger = Logger(log_file)
    #################
    # Data Iterator #
    #################
    data = Dataset(dataname, max_len, max_vocab, batch_size, logger)
    data.read_data()
    train_iter = data.data_generator(data.train)
    val_iter = data.data_generator(data.val)
    test_iter = data.data_generator(data.test)
    # batch_num = data.get_batch_num(train_iter, max_len, batch_size)
    # print(batch_num)

    encoder = Encoder(emb_size, gru_dim, latent_dim, len(data.char2id)).to(device)
    if scratch:
        decoder = Decoder(latent_dim, encoder.emb_layer, head_num, head_size).to(device)
    else:
        decoder = Decoder_torch(latent_dim, encoder.emb_layer, head_num, head_size, data.vocab_size).to(device)
    EnDecoder = EnDecoder_forward(encoder, decoder, data.pad_token)
    optimizer_encoder= torch.optim.SGD(encoder.parameters(), lr=1e-3, weight_decay=wd, momentum=0.9)
    optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=1e-3, weight_decay=wd, momentum=0.9)
    start_iter = 0
    best_val = 99999999
    if resume_file is not None:
        ckpt_enc = torch.load(f"./ckpt/{dataname}_mmd/enc/{resume_file}")
        encoder.load_state_dict(ckpt_enc['model'])
        start_iter = ckpt_enc['iter']
        optimizer_encoder.load_state_dict(ckpt_enc['optimizer'])
        ckpt_dec = torch.load(f"./ckpt/{dataname}_mmd/dec/{resume_file}")
        decoder.load_state_dict(ckpt_dec['model'])
        optimizer_decoder.load_state_dict(ckpt_dec['optimizer'])
    #########################
    # Training & Evaluation #
    #########################
    total_seq_loss = 0
    stop_sign = 0
    for iter in range(start_iter, start_iter + iterations):
        encoder.train()
        decoder.train()
        train_data = next(train_iter)[0]
        kl_loss, d_loss, ende_loss, seq_loss = EnDecoder(train_data.to(device))
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        if add_kl:
            ende_loss += kl_anneal_function(kl_anneal_func, iter, iterations) * max(kl_thresh, kl_loss)
        ende_loss.backward()
        nn.utils.clip_grad_norm_(EnDecoder.parameters(), clip)
        optimizer_encoder.step()
        optimizer_decoder.step()

        if iter % log_iter == 0:
            word_ppl = np.exp(seq_loss.item())
            logger.info('iter: {}, Train: EnDecoder = '
                  'weighted d_loss: {:5.9f}, g_loss: {:5.9f}, kl: {:5.9f} | ppl: {:5.9f}'
                  .format(iter, alpha * d_loss.item(), ende_loss.item(),
                          kl_loss.item(), word_ppl))
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                val_data = next(val_iter)[0]
                val_kl_loss, val_d_loss, val_ende_loss, val_seq_loss = EnDecoder(val_data.to(device))
                val_word_ppl = np.exp((val_seq_loss).item())
                logger.info('Evaluate: EnDecoder = '
                      'weighted d_loss: {:5.9f}, g_loss: {:5.9f}, kl: {:5.9f} | ppl: {:5.9f}'
                      .format(alpha * val_d_loss.item(),
                              val_ende_loss.item(), val_kl_loss.item(), val_word_ppl))
                sys.stdout.flush()
                val_ende_loss = val_ende_loss.item()
            if val_ende_loss <= best_val:
                best_val = val_ende_loss
                if store:
                    logger.info("saving weights with best val: {:.4f} at {} iteration".format(best_val, iter))
                    sys.stdout.flush()
                    state = {'model': encoder.state_dict(), 'optimizer': optimizer_encoder.state_dict(), 'iter': iter}
                    with open(get_savepath(0, "enc"), 'wb') as f:
                        torch.save(state, f)
                    state = {'model': decoder.state_dict(), 'optimizer': optimizer_decoder.state_dict(), 'iter': iter}
                    with open(get_savepath(0, "dec"), 'wb') as f:
                        torch.save(state, f)
            else:
                if stop_sign >= stop:
                    logger.info(f"Early stop at {iter} iteration..")
                    break
                else:
                    stop_sign += 1

        if (iter % 2000 == 0) and (iter != 0):
            data.gen(False, latent_dim, decoder)
            data.reconstruct(2, test_iter, encoder, decoder)
            for _n in range(2):
                vec = np.random.normal(size=(1, latent_dim))
                data.gen_bs(vec, latent_dim, decoder, topk=1)
            logger.info('diversity 0.8')
            sys.stdout.flush()
            data.interpolate(0.8, 5, test_iter, encoder, decoder)
            logger.info('diversity 1.0')
            sys.stdout.flush()
            data.interpolate(1.0, 5, test_iter, encoder, decoder)

if __name__=="__main__":
    train2()