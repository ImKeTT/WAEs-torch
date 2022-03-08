#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: WAE_MMD_2.py
@author: ImKe at 2021/12/24
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from logger import Logger
import tqdm

import numpy as np
import sys
import os
import datetime
import random

####################
# Hyper Parameters #
####################
max_len = 20
max_vocab = 30000

emb_size = 256
gru_dim = 150
batch_size = 512
latent_dim = 64
sigma = 1 ## variance of hidden dimension (default: 1)
hidden_dim = 128
# nambda = 20
head_num = 8
nlayers = 1 ## GRU decoder layer number

iterations = 100000 # total training iteration
epochs = 10
max_batch_iter = 15000
log_iter = 200
alpha = 10.0 # weight of VAE normalization term
lambda_gp = 10.0 # weight of Gradient Penalty
p = 6 # for gradient computation
k = 2 # for gradient computation
wd = 1e-5 # weight decay of optimizer
dataname = "apnews" # data name
kernel = "RBF" # kernel method for MMD
imq_c = 1 # parameter c for IMQ kernel method (default 1)
seed = 42
lr = 0.0001
stop = 1000
add_kl = True
kl_anneal_func = "const" # work when add_kl is True
kl_thresh = 0.0
store = False

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
    ckpt_root = f"./ckpt_2/{dataname}_mmd/{mode}"
    os.makedirs(ckpt_root, exist_ok=True)
    path = f"{ckpt_root}/iter{iter_n}-emb{emb_size}.gru{gru_dim}.bs{batch_size}.latent{latent_dim}." \
           f"hiddim{hidden_dim}.alpha{alpha}.kl.{add_kl}_{kl_anneal_func}.thresh{kl_thresh}.{dataname}.date{now.month}-{now.day}.pt"
    return path

##########
# Models #
##########
class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.dis_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            # nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z):
        return self.dis_net(z)

class Encoder(nn.Module):
    """
    Sequence Encoder (Transformer)
    """
    def __init__(self, emb_size, head_n, hidden_dim, latent_dim, vocab_size, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.position_embed = PositionalEncoding(emb_size)
        encoder_layers = nn.TransformerEncoderLayer(emb_size, head_n, dropout=dropout)
        self.transformer_decoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.emb2hidden = nn.Sequential(nn.Linear(emb_size, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim))
        self.fclv = nn.Linear(hidden_dim, latent_dim)
        self.fcmu = nn.Linear(hidden_dim, latent_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reparameterize(self, mean, logvar):
        sd = torch.exp(0.5 * logvar)  # Standard deviation
        # We assume the posterior is a multivariate Gaussian
        eps = torch.randn_like(sd)
        z = eps.mul(sd).add(mean)
        return z

    def forward(self, encoder_input):
        encoder_emb = self.emb(encoder_input).transpose(0, 1) ## [max_len, bs, emb_size]
        encoder_emb = self.position_embed(encoder_emb)
        encoder_repr = self.transformer_decoder(encoder_emb) ## [max_len, bs, emb_size]
        ## use the last representation of transformer to parameterize mu and lv
        encoder_repr = encoder_repr.transpose(0, 1)[:, -1, :].squeeze(1) ## [bs, emb_size]
        encoder_repr = self.emb2hidden(encoder_repr)
        lv = self.fclv(encoder_repr)
        mu = self.fcmu(encoder_repr)
        sample = self.reparameterize(mu, lv)
        return mu, lv, sample


class Decoder(nn.Module):
    def __init__(self, embed_dim, gru_dim, nlayers, latent_dim, vocab_size, emb_layer, dropout=0.1, batch_first=True):
        super().__init__()
        self.hidden_size = gru_dim
        self.embed_size = embed_dim
        self.nlayers = nlayers
        self.emb = emb_layer

        self.decoder = nn.GRU(embed_dim, gru_dim, nlayers, dropout=dropout, batch_first=batch_first)
        self.final_linear = nn.Linear(gru_dim, vocab_size)
        self.z2hidden = nn.Linear(latent_dim, gru_dim)

    def forward(self, dec_in, z, max_len):
        bs = z.size(0)
        h_init = torch.zeros([self.nlayers, bs, self.hidden_size]).to(z.device)
        h_init[0, :, :] = self.z2hidden(z)
        dec_in = self.emb(dec_in)
        outputs, _ = self.decoder(dec_in, h_init)
        logits = self.final_linear(outputs)
        return logits

#######################################
# Training functions for WAE with MMD #
#######################################
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

        dec_out = self.decoder(dec_in, q_z, max_len)
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

#########################
# ======== WAE ======== #
#########################
def train(resume_file=None):
    log_dir = f"./log_2/{dataname}_mmd"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/emb{emb_size}.gru{gru_dim}.bs{batch_size}.latent{latent_dim}." \
           f"hiddim{hidden_dim}.alpha{alpha}.kl.{add_kl}_{kl_anneal_func}.thresh{kl_thresh}.{dataname}.date{now.month}-{now.day}.txt"
    logger = Logger(log_file)
    #################
    # Data Iterator #
    #################
    data = Dataset(dataname, max_len, max_vocab, batch_size, logger)
    data.read_data()
    train_iter = data.data_generator(data.train)
    val_iter = data.data_generator(data.val)
    test_iter = data.data_generator(data.test)

    discriminator = Discriminator(latent_dim, hidden_dim).to(device)
    encoder = Encoder(emb_size, head_num, hidden_dim, latent_dim, data.vocab_size).to(device)
    decoder = Decoder(emb_size, gru_dim, nlayers, latent_dim, data.vocab_size, encoder.emb).to(device)
    EnDecoder = EnDecoder_forward(encoder, decoder, data.pad_token)
    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=5e-3, weight_decay=wd)
    optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=5e-3, weight_decay=wd)
    start_iter = 0
    best_val = 99999999
    if resume_file is not None:
        start_iter = ckpt_dis['iter']
        ckpt_enc = torch.load(f"./ckpt_2/{dataname}/enc/{resume_file}")
        encoder.load_state_dict(ckpt_enc['model'])
        optimizer_encoder.load_state_dict(ckpt_enc['optimizer'])
        ckpt_dec = torch.load(f"./ckpt_2/{dataname}/dec/{resume_file}")
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
        kl_loss, d_loss, errG, seq_loss = EnDecoder(train_data.to(device))
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        if add_kl:
            errG += kl_anneal_function(kl_anneal_func, iter, iterations)* max(kl_thresh, kl_loss)
        errG.backward()
        nn.utils.clip_grad_norm_(EnDecoder.parameters(), 5.0)
        optimizer_encoder.step()
        optimizer_decoder.step()
        total_seq_loss += seq_loss.item()

        if iter % log_iter == 0:
            word_ppl = np.exp(seq_loss.item())
            logger.info('iter: {}, Train: EnDecoder = '
                  'd_loss: {:5.9f}, errG: {:5.9f}, kl: {:5.9f} | ppl: {:5.9f}'
                  .format(iter, d_loss.item(), errG.item(),
                          kl_loss.item(), word_ppl))
            total_seq_loss = 0
            discriminator.eval()
            EnDecoder.eval()
            z_sample = torch.randn(size=(batch_size, latent_dim)).to(device)
            # val_w_dist, val_dis_loss = D_forward(discriminator, encoder, decoder,
            #                              next(val_iter)[0][0].to(device), z_sample)
            val_data = next(val_iter)[0]
            val_kl_loss, val_d_loss, val_errG, val_seq_loss = EnDecoder(val_data.to(device))
            val_word_ppl = np.exp((val_seq_loss).item())
            logger.info('Evaluate:  EnDecoder = '
                  'd_loss: {:5.9f}, errG: {:5.9f}, kl: {:5.9f} | ppl: {:5.9f}'
                  .format(val_d_loss.item(),
                          val_errG.item(), val_kl_loss.item(), val_word_ppl))
            sys.stdout.flush()
            val_errG = val_errG.item()
            if val_errG <= best_val:
                best_val = val_errG
                if store:
                    logger.info("saving weights with best generator val error: {:.4f} at {} iteration".format(best_val, iter))
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
    if store:
        logger.info("saving weights at the last iteration...")
        state = {'model': encoder.state_dict(), 'optimizer': optimizer_encoder.state_dict(), 'iter': iter}
        with open(get_savepath(iter, "enc"), 'wb') as f:
            torch.save(state, f)
        state = {'model': decoder.state_dict(), 'optimizer': optimizer_decoder.state_dict(), 'iter': iter}
        with open(get_savepath(iter, "dec"), 'wb') as f:
            torch.save(state, f)

        # if (iter % 5000 == 0) and (iter != 0):
        #     print("saving model weights")
        #     sys.stdout.flush()
        #     state = {'model': discriminator.state_dict(), 'optimizer': optimizer_dis.state_dict(), 'iter': iter}
        #     with open(get_savepath(iter, "dis"), 'wb') as f:
        #         torch.save(state, f)
        #     state = {'model': encoder.state_dict(), 'iter': iter}
        #     with open(get_savepath(iter, "enc"), 'wb') as f:
        #         torch.save(state, f)
        #     state = {'model': decoder.state_dict(), 'optimizer': optimizer_en_decoder.state_dict(), 'iter': iter}
        #     with open(get_savepath(iter, "dec"), 'wb') as f:
        #         torch.save(state, f)

if __name__=="__main__":
    train()