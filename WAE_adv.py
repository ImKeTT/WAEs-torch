#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: WAE_adv.py
@author: ImKe at 2021/8/13
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

from utils import *
from logger import Logger

torch.set_printoptions(precision=15)

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
## Note that emb_size + n*latent_dim should be divided by head_num
head_size = [(emb_size + latent_dim) // head_num,
             (emb_size + 2*latent_dim) // head_num,
             (emb_size + 3*latent_dim) // head_num] # for self attention
iterations = 200000 # total training iteration
epochs = 10
max_batch_iter = 15000
log_iter = 200
alpha = 10.0 # weight of VAE normalization term
lambda_gp = 10.0 # weight of Gradient Penalty
p = 6 # for gradient computation
k = 2 # for gradient computation
wd = 1e-5 # weight decay of optimizer
dataname = "apnews" # data name
seed = 42
lr = 0.0001
stop = 500
add_kl = True # whether add KLD as regularization
kl_anneal_func = None # work when add_kl is True, "linear" or None or "const"
kl_thresh = 0.05
scratch = False # use transformer implemented from scratch
store = False # whether to store checkpoint for model training

torch.manual_seed(seed)
random.seed(seed)
now = datetime.datetime.now()
device = "cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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
    Sequence Encoder (GRU)
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
        # We assume the posterior is a multivariate Gaussian
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
        self.embed = embed_layer
        self.position_embed = PositionalEncoding(emb_size)
        self.head_n = head_n
        self.head_size = head_size
        self.z2src = nn.Linear(latent_dim, emb_size)
        self.latent_dim = latent_dim
        ## torch v1.4.0 does not support batch_first=True
        decoder_layers = nn.TransformerDecoderLayer(emb_size, head_n, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=3)
        self.final_linear = nn.Linear(emb_size, vocab_size)

    @staticmethod
    def autoregressive_mask(tensor):
        """Generate auto - regressive mask for tensor. It's used to preserving the auto - regressive property.
        Args:
            tensor(torch.Tensor): of shape ``(batch, seq_len)``.
        Returns:
            torch.Tensor: a byte mask tensor of shape ``(batch, seq_len, seq_len)`` containing mask for
            illegal attention connections between decoder sequence tokens.
        """
        batch_size, seq_len = tensor.shape
        x = torch.ones(
            seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

        return x.repeat(batch_size, 1, 1).byte()

    def forward(self, dec_in, enc_z, max_len=max_len):
        src = self.embed(dec_in) ## [bs, max_len, emb_size]
        src = src.transpose(0, 1) ## [max_len, bs, emb_size]
        src = self.position_embed(src) ## [max_len, bs, emb_size]
        enc_z = self.z2src(enc_z).unsqueeze(0).repeat(max_len, 1, 1) ## [max_len, bs, emb_size]
        assert enc_z.size(-1)==src.size(-1)
        # mask = self.autoregressive_mask(dec_in)
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
        decoder_output = self.dec_softmax(decoder_h) ## [bs, max_len, vocab_size]
        return decoder_output

class Decoder_(nn.Module):
    """
    Sequence Decoder
    less elegant way for transformer decoder from scratch
    """
    def __init__(self, latent_dim, embed_layer, head_n, head_size):
        super().__init__()
        # self.activation = nn.Softmax()
        self.activation = None # nn.CrossEntropy() without softmax = K.categorical_crossentropy()
        self.dec_softmax = TiedEmbeddingsTransposed(embed_layer, self.activation)
        self.embed = embed_layer
        self.position_embed = PositionalEmbedding()
        self.decoder_net = nn.ModuleList()
        self.head_n = head_n
        self.head_size = head_size
        # self.layer_norm = LayerNormalization(latent_dim * 3)
        self.latent_dim = latent_dim
        self.act2 = nn.ReLU()

    def forward(self, dec_in, enc_z, max_len):
        decoder_z = enc_z.unsqueeze(1).repeat(1, max_len, 1)
        decoder_embed = self.embed(dec_in)
        decoder_h = self.position_embed(decoder_embed)
        print("0", decoder_z.size(), decoder_h.size())
        for layer in range(3):
            print(layer)
            dense_layer = nn.Linear(self.latent_dim, self.latent_dim).to(device)
            decoder_z_hier = dense_layer(decoder_z)
            decoder_h = torch.cat([decoder_h, decoder_z_hier], -1)
            print("1", decoder_h.size())
            att_layer = Attention(self.head_n, self.head_size[layer], max_len,
                                       [decoder_h.size() for _ in range(3)]).to(device)
            decoder_h_attn = att_layer([decoder_h, decoder_h, decoder_h])
            decoder_h = torch.add(decoder_h,decoder_h_attn)
            print("2",decoder_h.size())
            layer_norm = LayerNormalization(decoder_h.size(-1)).to(device)
            decoder_h = layer_norm(decoder_h)
            print("3", decoder_h.size())
            decoder_h_mlp = nn.ReLU()(nn.Linear(decoder_h.size(-1),
                                              self.head_size[layer] * head_num).to(device)(decoder_h))
            decoder_h = torch.add(decoder_h, decoder_h_mlp)
            decoder_h = layer_norm(decoder_h)
            decoder_h = self.position_embed(decoder_h)
            print("4", decoder_h.size())
        decoder_h = nn.Linear(decoder_h.size(-1), decoder_embed.size(-1)).to(device)(decoder_h)
        # decoder_h = self.final_linear(decoder_h)
        decoder_output = self.dec_softmax(decoder_h)
        return decoder_output

######################
# Training functions #
######################
class Dis_forward_prev(nn.Module):
    def __init__(self, discriminator, encoder, decoder):
        super().__init__()
        self.dis = discriminator
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_in, z_in):
        bs = enc_in.size(0) # batch size
        ########################
        # Discriminator Output #
        ########################
        mu, lv, z_sample = self.encoder(enc_in)
        z_fake = z_sample

        z_in.requires_grad = True
        z_fake.requires_grad = True
        z_real_score = self.dis(z_in)
        z_fake_score = self.dis(z_fake)
        dummy_real = torch.ones((bs, 1), requires_grad=False).to(device)
        dummy_fake = torch.ones((bs, 1), requires_grad=False).to(device)
        real_grad = torch.autograd.grad(z_real_score, z_in, grad_outputs=dummy_real, create_graph=True, only_inputs=True)[0]
        fake_grad = torch.autograd.grad(z_fake_score, z_fake, grad_outputs=dummy_fake, create_graph=True, only_inputs=True)[0]

        d_loss = torch.mean(z_fake_score - z_real_score)
        real_grad_norm = torch.sum(real_grad ** 2, axis=1) ** (p / 2)
        fake_grad_norm = torch.sum(fake_grad ** 2, axis=1) ** (p / 2)
        grad_loss = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

        w_dist = torch.abs(torch.mean(z_real_score - z_fake_score))
        dis_model_loss = torch.abs(d_loss - grad_loss)
        # dis_model_loss = d_loss
        return w_dist, dis_model_loss

class Dis_forward(nn.Module):
    def __init__(self, discriminator, encoder, decoder):
        super().__init__()
        self.dis = discriminator
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_in, z_real):
        bs = enc_in.size(0) # batch size
        ########################
        # Discriminator Output #
        ########################
        mu, lv, z_sample = self.encoder(enc_in)
        z_fake = z_sample
        ######################
        # Discriminator Loss #
        ######################
        z_real_score = self.dis(z_real)
        z_fake_score = self.dis(z_fake)
        dis_loss = -torch.mean(z_real_score) + torch.mean(z_fake_score)
        ####################
        # Gradient Penalty #
        ####################
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn(bs, 1, device=device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * z_real + ((1 - alpha) * z_fake)).requires_grad_(True)
        d_interpolates = self.dis(interpolates)
        fake = torch.full((bs, 1), 1, device=device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        return dis_loss, gradient_penaltys

class EnDecoder_forward(nn.Module):
    def __init__(self, discriminator, encoder, decoder, ignore_index):
        super().__init__()
        self.dis = discriminator
        self.encoder = encoder
        self.decoder = decoder
        self.ii = ignore_index

    def forward(self, data_iter):
        enc_in = data_iter[0]
        dec_in = data_iter[1]
        dec_true = data_iter[2]
        mu, lv, enc_z = self.encoder(enc_in)
        kl_loss = torch.mean(- 0.5 * torch.sum(1 + lv - mu**2 - torch.exp(lv), -1), -1)
        z_fake_score = self.dis(self.encoder(enc_in.data)[0]) ## use mean for discriminator calculating fake score

        dec_out = self.decoder(dec_in, enc_z)
        dec_true_mask = (dec_true.unsqueeze(2)>0).float()
        # nn.CrossEntropyLoss() has the reverse position w.r.t. label and output regard K.catgorical_crossentrophy
        xent_loss = torch.sum(F.cross_entropy(dec_out.contiguous().view(-1, dec_out.size(-1)), dec_true.view(-1),
                                              ignore_index=self.ii
                                              ) * dec_true_mask[:,:,0] / torch.sum(dec_true_mask[:,:,0]))
        d_loss = torch.mean(z_fake_score)
        # xent_loss /= batch_size * max_len
        ## Generator loss
        all_loss = xent_loss + alpha * d_loss

        return kl_loss, d_loss, all_loss, xent_loss

#############################
# ======== WAE-adv ======== #
#############################
def criterion(reconst, target, pad_token):
    return F.cross_entropy(reconst.view(-1, reconst.size(2)),
                           target.view(-1), ignore_index=pad_token)

#################
# Training Main #
#################
def train(resume_file=None):
    log_dir = f"/data/tuhq/toemlm/WAE/log/{dataname}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/emb{emb_size}.gru{gru_dim}.bs{batch_size}.latent{latent_dim}." \
               f"hiddim{hidden_dim}.alpha{alpha}.kl.{add_kl}_{kl_anneal_func}.thresh{kl_thresh}." \
               f"scratch.{scratch}.{dataname}.date{now.month}-{now.day}.txt"
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

    discriminator = Discriminator(latent_dim, hidden_dim).to(device)
    encoder = Encoder(emb_size, gru_dim, latent_dim, len(data.char2id)).to(device)
    if scratch:
        decoder = Decoder(latent_dim, encoder.emb_layer, head_num, head_size).to(device)
    else:
        decoder = Decoder_torch(latent_dim, encoder.emb_layer, head_num, head_size, data.vocab_size).to(device)

    # Optimizers
    enc_optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr)
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr=0.5 * lr)

    # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=30, gamma=0.5)
    # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=30, gamma=0.5)
    # dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size=30, gamma=0.5)

    ## resume if available
    if resume_file is not None:
        ckpt_dis = torch.load(f"./ckpt/{dataname}/dis/{resume_file}")
        discriminator.load_state_dict(ckpt_dis['model'])
        dis_optim.load_state_dict(ckpt_dis['optimizer'])
        start_epoch = ckpt_dis['epoch']
        ckpt_enc = torch.load(f"./ckpt/{dataname}/enc/{resume_file}")
        encoder.load_state_dict(ckpt_enc['model'])
        ckpt_dec = torch.load(f"./ckpt/{dataname}/dec/{resume_file}")
        decoder.load_state_dict(ckpt_dec['model'])
        enc_optim.load_state_dict(ckpt_dec['optimizer'])

    for epoch in range(start_epoch, start_epoch + epochs):
        iter = 0
        stop_sign = 0
        encoder.train()
        decoder.train()
        discriminator.train()
        for inum, text in enumerate(train_iter):
            # text = text.to(device)
            enc_in = text[0][0].to(device)
            dec_in = text[0][1].to(device)
            text_true = text[0][2].to(device)

            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()

            # ======== Train Discriminator ======== #

            frozen_params(decoder)
            frozen_params(encoder)
            free_params(discriminator)

            z_fake = torch.randn(batch_size, latent_dim) * sigma

            if torch.cuda.is_available():
                z_fake = z_fake.cuda()

            d_fake = discriminator(z_fake)

            mu, logvar, z_real = encoder(enc_in)
            d_real = discriminator(z_real)

            dis_loss = -(torch.log(d_fake).mean() + torch.log(1 - d_real).mean())
            dis_loss.backward()
            # torch.log(d_fake).mean().unsqueeze(0).backward(mone)
            # torch.log(1 - d_real).mean().unsqueeze(0).backward(mone)

            dis_optim.step()

            # ======== Train Generator ======== #

            free_params(decoder)
            free_params(encoder)
            frozen_params(discriminator)


            mu, logvar, z_real = encoder(enc_in)
            x_recon = decoder(dec_in, z_real)
            d_real = discriminator(encoder(enc_in.data)[0])

            recon_loss = criterion(x_recon, text_true, data.pad_token)
            d_loss = -alpha * (torch.log(d_real)).mean()
            kl_loss = torch.mean(- 0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), -1), -1)

            # recon_loss.unsqueeze(0).backward(one)
            # d_loss.unsqueeze(0).backward(mone)
            endecoder_loss = recon_loss + d_loss
            endecoder_loss.backward()
            # kl_loss = torch.mean(- 0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), -1), -1)

            enc_optim.step()
            dec_optim.step()

            iter += 1

            if (iter+1) % 200 == 0:
                logger.info("Train: Epoch: [%d/%d], Iter %d, Rec Loss: %.4f, Dis Loss: %.4f, KL Loss: %.4f" %
                      (epoch + 1, epochs, iter + 1, recon_loss/batch_size, d_loss, kl_loss))
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
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        test_data = next(test_iter)[0]
        # test_data = test_data.to(device)
        test_enc_in = test_data[0].to(device)
        test_dec_in = test_data[1].to(device)
        test_text_true = test_data[2].to(device)

        test_mu, test_logvar, z_test_real = encoder(test_enc_in)
        x_test_recon = decoder(test_dec_in, z_test_real, max_len)
        d_test_real = discriminator(z_test_real)

        recon_loss = criterion(x_test_recon, test_text_true, data.pad_token)
        d_loss = alpha * (torch.log(d_test_real)).mean()

        logger.info("Test: Epoch: [%d/%d], Rec Loss: %.4f, Dis Loss: %.4f" %
              (epoch + 1, epochs, recon_loss, d_loss))


#################################
# ======== WAE with gp ======== #
#################################
def train2(resume_file=None):
    log_dir = f"./log/{dataname}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/emb{emb_size}.gru{gru_dim}.bs{batch_size}.latent{latent_dim}." \
           f"hiddim{hidden_dim}.alpha{alpha}.kl.{add_kl}_{kl_anneal_func}.thresh{kl_thresh}." \
               f"scratch.{scratch}.{dataname}.date{now.month}-{now.day}.txt"
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
    encoder = Encoder(emb_size, gru_dim, latent_dim, len(data.char2id)).to(device)
    if scratch:
        decoder = Decoder(latent_dim, encoder.emb_layer, head_num, head_size).to(device)
    else:
        decoder = Decoder_torch(latent_dim, encoder.emb_layer, head_num, head_size, data.vocab_size).to(device)
    EnDecoder = EnDecoder_forward(discriminator, encoder, decoder, data.pad_token)
    Dis_f = Dis_forward(discriminator, encoder, decoder)
    optimizer_dis = torch.optim.SGD(discriminator.parameters(), lr=1e-3, weight_decay=wd, momentum=0.9)
    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=1e-3, weight_decay=wd, momentum=0.9)
    optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=1e-3, weight_decay=wd, momentum=0.9)
    start_iter = 0
    best_val = 99999999
    if resume_file is not None:
        ckpt_dis = torch.load(f"./ckpt/{dataname}/dis/{resume_file}")
        discriminator.load_state_dict(ckpt_dis['model'])
        optimizer_dis.load_state_dict(ckpt_dis['optimizer'])
        start_iter = ckpt_dis['iter']
        ckpt_enc = torch.load(f"./ckpt/{dataname}/enc/{resume_file}")
        encoder.load_state_dict(ckpt_enc['model'])
        optimizer_encoder.load_state_dict(ckpt_enc['optimizer'])
        ckpt_dec = torch.load(f"./ckpt/{dataname}/dec/{resume_file}")
        decoder.load_state_dict(ckpt_dec['model'])
        optimizer_decoder.load_state_dict(ckpt_dec['optimizer'])
    #########################
    # Training & Evaluation #
    #########################
    # training scale 3 : 1
    total_seq_loss = 0
    stop_sign = 0
    for iter in range(start_iter, start_iter + iterations):
        encoder.train()
        decoder.train()
        discriminator.train()
        train_data = next(train_iter)[0]
        for _ in range(2):
            frozen_params(decoder)
            frozen_params(encoder)
            free_params(discriminator)
            z_sample = torch.randn(size=(batch_size, latent_dim)).to(device)
            D_loss, gp_loss = Dis_f(train_data[0].to(device), z_sample)
            # w_dist, dis_loss = D_forward(discriminator, encoder, decoder,
            #                              next(train_iter)[0][0].to(device), z_sample)
            errD = D_loss + gp_loss
            optimizer_dis.zero_grad()
            errD.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            optimizer_dis.step()
        for _ in range(1):
            free_params(decoder)
            free_params(encoder)
            frozen_params(discriminator)
            kl_loss, d_loss, errG, seq_loss = EnDecoder(train_data.to(device))
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            if add_kl:
                errG += kl_anneal_function(kl_anneal_func, iter, iterations) * max(kl_thresh, kl_loss)
            errG.backward()
            nn.utils.clip_grad_norm_(EnDecoder.parameters(), 5.0)
            optimizer_encoder.step()
            optimizer_decoder.step()
        total_seq_loss += seq_loss.item()

        if iter % log_iter == 0:
            word_ppl = np.exp(seq_loss.item())
            logger.info('iter: {}, Train: Dis = errD: {:5.9f}, gp loss: {:5.9f} | EnDecoder = '
                  'd_loss: {:5.9f}, errG: {:5.9f}, kl: {:5.9f} | ppl: {:5.9f}'
                  .format(iter, D_loss.item(), gp_loss.item(), d_loss.item(), errG.item(),
                          kl_loss.item(), word_ppl))
            total_seq_loss = 0
            discriminator.eval()
            encoder.eval()
            decoder.eval()
            z_sample = torch.randn(size=(batch_size, latent_dim)).to(device)
            # val_w_dist, val_dis_loss = D_forward(discriminator, encoder, decoder,
            #                              next(val_iter)[0][0].to(device), z_sample)
            val_data = next(val_iter)[0]
            val_D_loss, val_gp = Dis_f(val_data[0].to(device), z_sample)
            val_errD = val_D_loss + val_gp
            val_kl_loss, val_d_loss, val_errG, val_seq_loss = EnDecoder(val_data.to(device))
            val_word_ppl = np.exp((val_seq_loss).item())
            logger.info('Evaluate: Dis = errD: {:5.9f}, gp loss: {:5.9f} | EnDecoder = '
                  'd_loss: {:5.9f}, errG: {:5.9f}, kl: {:5.9f} | ppl: {:5.9f}'
                  .format(val_errD.item(), val_gp.item(), val_d_loss.item(),
                          val_errG.item(), val_kl_loss.item(), val_word_ppl))
            sys.stdout.flush()
            val_errG = val_errG.item()
            if val_errG <= best_val:
                best_val = val_errG
                if store:
                    logger.info("saving weights with best val: {:.4f} at {} iteration".format(best_val, iter))
                    sys.stdout.flush()
                    state = {'model': discriminator.state_dict(), 'optimizer': optimizer_dis.state_dict(), 'iter': iter}
                    with open(get_savepath(0, "dis"), 'wb') as f:
                        torch.save(state, f)
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
            data.gen(False, latent_dim, decoder)
            data.reconstruct(2, test_iter, encoder, decoder)
            for _n in range(2):
                vec = np.random.normal(size=(1, latent_dim))
                data.gen_bs(vec, latent_dim, decoder, topk=1)
            logger.info('diversity 0.8')
            sys.stdout.flush()
            data.interpolate(0.8, 8, test_iter, encoder, decoder)
            logger.info('diversity 1.0')
            sys.stdout.flush()
            data.interpolate(1.0, 8, test_iter, encoder, decoder)
    if store:
        state = {'model': discriminator.state_dict(), 'optimizer': optimizer_dis.state_dict(), 'iter': iter}
        with open(get_savepath(iter, "dis"), 'wb') as f:
            torch.save(state, f)
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
    train2()