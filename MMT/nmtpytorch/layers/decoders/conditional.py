# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import torch.autograd as autograd
from ...utils.nn import get_rnn_hidden_state
from .. import FF
from ..variational_dropout import VariationalDropout
from .adversarial import Generator_adv,Discriminator_adv
import matplotlib.pyplot as plt
import os
import pickle
from ..attention import Attention, HierarchicalAttention
from .. import Fusion


class ConditionalDecoder(nn.Module):
    """A conditional decoder with attention à la dl4mt-tutorial."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, aux_ctx_name, n_vocab,
                 rnn_type, tied_emb=False, dec_init='zero', dec_init_activ='tanh',
                 dec_init_size=None, att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 transform_ctx=True, mlp_bias=False, dropout_out=0, dropout_emb=0,
                 dropout_dec=0, variational_dropout=False,
                 emb_maxnorm=None, emb_gradscale=False,
                 imagination=False,
                 loss_imagination=False,
                 imagination_factor=0,
                 gradient_penality=0,
                 critic=0,
                 image_att=False,
                 fusion_type=False,
                 ):
        super().__init__()

        # Normalize case
        self.rnn_type = rnn_type.upper()

        # Safety checks
        assert self.rnn_type in ('GRU', 'LSTM'), \
            "rnn_type '{}' not known".format(rnn_type)
        assert dec_init in ('zero', 'mean_ctx', 'feats'), \
            "dec_init '{}' not known".format(dec_init)

        RNN = getattr(nn, '{}Cell'.format(self.rnn_type))
        # LSTMs have also the cell state
        self.n_states = 1 if self.rnn_type == 'GRU' else 2

        # Set custom handlers for GRU/LSTM
        if self.rnn_type == 'GRU':
            self._rnn_unpack_states = lambda x: x
            self._rnn_pack_states = lambda x: x
        elif self.rnn_type == 'LSTM':
            self._rnn_unpack_states = self._lstm_unpack_states
            self._rnn_pack_states = self._lstm_pack_states

        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.aux_ctx_name = aux_ctx_name
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.dec_init = dec_init
        self.dec_init_size = dec_init_size
        self.dec_init_activ = dec_init_activ
        self.att_type = att_type
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.mlp_bias = mlp_bias
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale

        self.imagination = imagination
        self.loss_imagination = loss_imagination
        self.imagination_factor = imagination_factor

        self.dropout_out = dropout_out
        self.dropout_dec = dropout_dec
        self.dropout_emb = dropout_emb
        self.variational_dropout = variational_dropout

        self.image_att = image_att
        self.fusion_type = fusion_type

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Create attention txt layer
        self.att = Attention(self.ctx_size_dict[self.ctx_name], self.hidden_size,
                             transform_ctx=self.transform_ctx,
                             mlp_bias=self.mlp_bias,
                             att_type=self.att_type,
                             att_activ=self.att_activ,
                             att_bottleneck=self.att_bottleneck,
                             temp=self.att_temp)

        # Create attention img layer
        if self.image_att:
            self.img_att_ff = FF(
                2048,
                self.hidden_size,
                activ='tanh')
            if fusion_type == "hierarchical":
                self.fusion = HierarchicalAttention(
                    [self.hidden_size, self.hidden_size],
                    self.hidden_size, self.hidden_size)
            else:
                self.fusion = Fusion(
                    fusion_type, 2 * self.hidden_size, self.hidden_size)



        if self.imagination:
            if self.loss_imagination == "cos" or self.loss_imagination == "acos":
                self.img_pool = FF(
                    self.hidden_size,
                    2048, activ='tanh')


            if "adv" in self.loss_imagination:
                self.g_stats = []
                self.d_stats = []
                self.i_stats = []
                self.cur_epoch = -1

                self.generator = Generator_adv(self.hidden_size)
                self.discriminator = Discriminator_adv(self.hidden_size)
                # self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                if self.loss_imagination == "adv":
                    self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
                if "adv_gp" in self.loss_imagination:
                    self.gradient_penality = gradient_penality
                    self.critic = critic
                    print("gradient_penality",gradient_penality,"critic",critic)
                    self.cur_g_loss = torch.Tensor([0])
                    self.cur_d_loss = torch.Tensor([0])


            if "waae" in self.loss_imagination:
                self.g_stats = []
                self.d_stats = []
                self.i_stats = []
                self.r_stats = []
                self.cur_epoch = -1

                self.img_pool = FF(
                    self.hidden_size,
                    2048, activ='tanh')
                self.discriminator = nn.Sequential(
                    nn.Linear(self.hidden_size, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1),
                    )
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
                self.z_var = 2
                self.critic = critic
                self.gradient_penality = gradient_penality
                print("gradient_penality", gradient_penality, "critic", critic)

        # Decoder initializer FF (for 'mean_ctx' or auxiliary 'feats')
        if self.dec_init in ('mean_ctx', 'feats'):
            if self.dec_init == 'mean_ctx':
                self.dec_init_size = self.ctx_size_dict[self.ctx_name]
            self.ff_dec_init = FF(
                self.dec_init_size,
                self.hidden_size * self.n_states, activ=self.dec_init_activ)

        # Create first decoder layer necessary for attention
        self.dec0 = RNN(self.input_size, self.hidden_size)
        self.dec1 = RNN(self.hidden_size, self.hidden_size)

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        if self.dropout_emb > 0:
            self.do_emb = nn.Dropout(self.dropout_emb)

        if self.dropout_dec > 0:
            self.do_dec = nn.Dropout(self.dropout_dec)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(self.hidden_size, self.input_size,
                          bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

        self.nll_loss = nn.NLLLoss(size_average=False, ignore_index=0)
        self.ic = 0

    def _lstm_pack_states(self, h):
        return torch.cat(h, dim=-1)

    def _lstm_unpack_states(self, h):
        # Split h_t and c_t into two tensors and return a tuple
        return torch.split(h, self.hidden_size, dim=-1)

    def _rnn_init_zero(self, ctx_dict):
        ctx, _ = ctx_dict[self.ctx_name]
        h_0 = torch.zeros(ctx.shape[1], self.hidden_size * self.n_states)
        return Variable(h_0).cuda()

    def _rnn_init_mean_ctx(self, ctx_dict):
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        if ctx_mask is None:
            return self.ff_dec_init(ctx.mean(0))
        else:
            return self.ff_dec_init(ctx.sum(0) / ctx_mask.sum(0).unsqueeze(1))

    def _rnn_init_feats(self, ctx_dict):
        ctx, _ = ctx_dict['feats']
        return self.ff_dec_init(ctx)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.alphas = []
        return self._init_func(ctx_dict)

    def f_next(self, ctx_dict, y, h):
        # Get hidden states from the first decoder (purely cond. on LM)
        h1 = self.dec0(y, self._rnn_unpack_states(h))

        if self.dropout_dec > 0:
            h1 = self.do_dec(h1)


        # Apply attention
        self.txt_alpha_t, txt_z_t = self.att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])
        z_t = txt_z_t


        if self.image_att:
            #compute image attention
            img_z_t = self.img_att_ff(ctx_dict[self.aux_ctx_name][0])
            img_z_t = img_z_t.squeeze(0)
            #do fusion of txt and img att
            if self.fusion_type == "hierarchical":
                self.h_att, z_t = self.fusion([txt_z_t, img_z_t], h1.unsqueeze(0))
            else:
                z_t = self.fusion(txt_z_t, img_z_t)


        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2 = self.dec1(z_t, h1)

        if self.dropout_dec > 0:
            h2 = self.do_dec(h2)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, h2

    def forward(self, ctx_dict, y, kwargs):
        """Computes the softmax outputs given source annotations `ctxs` and
        ground-truth target token indices `y`. Only called during training.

        Arguments:
            ctxs(Variable): A variable of `S*B*ctx_dim` representing the source
                annotations in an order compatible with ground-truth targets.
            y(Variable): A variable of `T*B` containing ground-truth target
                token indices for the given batch.
        """
        loss = 0.0
        logps = None if self.training else torch.zeros(
            y.shape[0] - 1, y.shape[1], self.n_vocab).cuda()

        hs = None

        # Convert token indices to embeddings -> T*B*E
        y_emb = self.emb(y)
        if self.dropout_emb > 0:
            y_emb = self.do_emb(y_emb)

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0] - 1):
            log_p, h = self.f_next(ctx_dict, y_emb[t], h)
            if not self.training:
                logps[t] = log_p.data

            if hs is None:
                hs = h.unsqueeze(0)
            else:
                hs = torch.cat((hs,h.unsqueeze(0)), dim=0)

            loss += self.nll_loss(log_p, y[t + 1])

        if self.imagination:

            # if np.random.random_sample() < 0.1:
                image,_ = ctx_dict[self.aux_ctx_name]
                image = image.squeeze(0)
                gamma = self.imagination_factor

                if self.loss_imagination == "cos":
                    v = self.img_pool(hs.mean(0))
                    zero = Variable(torch.Tensor([0])).cuda()
                    alpha = Variable(torch.Tensor([0.1])).cuda()
                    loss += gamma * \
                            (torch.max(zero,alpha+F.cosine_similarity(v, image)
                                                 -F.cosine_similarity(v, image[torch.randperm(image.size()[0]).cuda()])
                                      ).sum()
                            )

                if self.loss_imagination == "acos":
                    v = self.img_pool(hs.mean(0))
                    loss += gamma * \
                            (
                                (1 - (torch.acos(F.cosine_similarity(v, image)) / math.pi)).sum()
                            )


                if self.loss_imagination == "adv":


                    Tensor = torch.cuda.FloatTensor
                    cond = hs.mean(0)

                    valid = Variable(Tensor(cond.size(0), 1).fill_(1.0), requires_grad=False)
                    fake = Variable(Tensor(cond.size(0), 1).fill_(0.0), requires_grad=False)

                    # z = Variable(Tensor(np.random.normal(0, 1, (cond.size(0), 100))))
                    gen_imgs = self.generator(cond)
                    d_input = torch.cat((gen_imgs,cond),dim=-1)
                    g_loss = gamma * self.adversarial_loss(self.discriminator(d_input), valid)
                    loss += g_loss

                    # Measure discriminator's ability to classify real from generated samples
                    if self.training: self.optimizer_D.zero_grad()

                    d_input1 = torch.cat((image,cond.detach()),dim=-1)
                    d_input2 = torch.cat((gen_imgs.detach(),cond.detach()),dim=-1)
                    real_loss = self.adversarial_loss(self.discriminator(d_input1), valid)
                    fake_loss = self.adversarial_loss(self.discriminator(d_input2), fake)
                    d_loss = gamma * ((real_loss + fake_loss) / 2)

                    if self.training:
                        d_loss.backward()
                        self.optimizer_D.step()
                    self.ic+=1

                    if self.ic % 30==0:
                        print("G_loss {:.3f} D_loss {:.3f}".format(float(g_loss.data), float(d_loss.data)))



                if self.loss_imagination == "adv_gp1":

                    Tensor = torch.cuda.FloatTensor
                    cond = hs.mean(0)


                    def compute_gradient_penalty(D, real_samples, fake_samples, cond):
                        """Calculates the gradient penalty loss for WGAN GP"""
                        # Random weight term for interpolation between real and fake samples
                        alpha = Tensor(np.random.random((real_samples.size(0), 1)))
                        # Get random interpolation between real and fake samples
                        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
                        interpolates = Variable(torch.cat((interpolates,cond), dim=-1), requires_grad=True)
                        d_interpolates = D(interpolates)
                        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
                        # Get gradient w.r.t. interpolates
                        gradients = autograd.grad(
                            outputs=d_interpolates,
                            inputs=interpolates,
                            grad_outputs=fake,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        return gradient_penalty

                    # ---------------------
                    if self.training:
                        self.optimizer_D.zero_grad()


                    # Generate a batch of images
                    z = Variable(Tensor(np.random.normal(0, 1, (cond.shape[0], 256))))
                    g_input = torch.cat((cond, z), dim=-1)

                    fake_imgs = self.generator(g_input).detach()

                    # Real images
                    d_input1 = torch.cat((image, cond.detach()), dim=-1)
                    d_input2 = torch.cat((fake_imgs, cond.detach()), dim=-1)
                    real_validity = self.discriminator(d_input1)
                    fake_validity = self.discriminator(d_input2)

                    # Gradient penalty
                    gradient_penalty = 0
                    if self.training:
                        gradient_penalty = compute_gradient_penalty(self.discriminator, image.data, fake_imgs.data, cond.data)
                    # Adversarial loss
                    d_loss = gamma * (-torch.mean(real_validity) + torch.mean(fake_validity) + self.gradient_penality * gradient_penalty)


                    if self.training:
                        d_loss.backward()
                        self.optimizer_D.step()


                    self.ic += 1

                    # Train the generator every n_critic steps
                    if self.ic % self.critic == 0:

                        # Sample noise as generator input
                        g_input = torch.cat((cond, z), dim=-1)
                        fake_imgs = self.generator(g_input)

                        # Loss measures generator's ability to fool the discriminator
                        # Train on fake images
                        d_input2 = torch.cat((fake_imgs, cond), dim=-1)
                        fake_validity = self.discriminator(d_input2)
                        g_loss = gamma * (-torch.mean(fake_validity))
                        loss += g_loss

                    if self.ic % 30==0:
                        print("G_loss {:.3f} D_loss {:.3f}".format(float(g_loss.data), float(d_loss.data)))
                        self.i_stats.append(self.ic)
                        self.g_stats.append(g_loss.data)
                        self.d_stats.append(d_loss.data)

                    if self.training:
                        if self.cur_epoch != kwargs["ectr"]:
                            self.cur_epoch = kwargs["ectr"]
                            plt.plot(self.i_stats, self.g_stats, '-r', label='G_loss')
                            plt.plot(self.i_stats, self.d_stats, '-b', label='D_loss')
                            plt.legend()
                            plt.savefig(os.path.join(kwargs["outdir"],'loss'+str(kwargs["seed"])+'.png'))
                            plt.clf()
                            pickle.dump([self.i_stats, self.g_stats, self.d_stats], open(os.path.join(kwargs["outdir"],'loss'+str(kwargs["seed"])+'.pkl'), 'wb+'))

                if self.loss_imagination == "adv_gp2":

                    Tensor = torch.cuda.FloatTensor
                    cond = hs.mean(0)
                    self.ic += 1


                    def compute_gradient_penalty(D, real_samples, fake_samples, cond):
                        """Calculates the gradient penalty loss for WGAN GP"""
                        # Random weight term for interpolation between real and fake samples
                        alpha = Tensor(np.random.random((real_samples.size(0), 1)))
                        # Get random interpolation between real and fake samples
                        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
                        interpolates = Variable(torch.cat((interpolates,cond), dim=-1), requires_grad=True)
                        d_interpolates = D(interpolates)
                        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
                        # Get gradient w.r.t. interpolates
                        gradients = autograd.grad(
                            outputs=d_interpolates,
                            inputs=interpolates,
                            grad_outputs=fake,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        return gradient_penalty


                    for _ in range(self.critic):

                        cond_D = cond.detach()

                        if self.training:
                            self.optimizer_D.zero_grad()

                        # Generate a batch of images
                        z = Variable(Tensor(np.random.normal(0, 1, (cond.shape[0], 256))))
                        g_input = torch.cat((cond_D, z), dim=-1)
                        fake_imgs = self.generator(g_input).detach()

                        # Real images
                        d_input1 = torch.cat((image, cond_D), dim=-1)
                        d_input2 = torch.cat((fake_imgs, cond_D), dim=-1)
                        real_validity = self.discriminator(d_input1)
                        fake_validity = self.discriminator(d_input2)

                        # Gradient penalty
                        gradient_penalty = 0
                        if self.training:
                            gradient_penalty = compute_gradient_penalty(self.discriminator, image.data, fake_imgs.data, cond.data)
                        # Adversarial loss
                        d_loss = gamma * (-torch.mean(real_validity) + torch.mean(fake_validity) + self.gradient_penality * gradient_penalty)

                        if self.training:
                            d_loss.backward(retain_graph=True)
                            self.optimizer_D.step()


                    # Sample noise as generator input
                    self.generator.zero_grad()
                    g_input = torch.cat((cond, z), dim=-1)
                    fake_imgs = self.generator(g_input)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    d_input2 = torch.cat((fake_imgs, cond), dim=-1)
                    fake_validity = self.discriminator(d_input2)
                    g_loss = gamma * (-torch.mean(fake_validity))
                    loss += g_loss


                    #stats

                    if self.ic % 30==0:
                        print("G_loss {:.3f} D_loss {:.3f}".format(float(g_loss.data), float(d_loss.data)))
                        self.i_stats.append(self.ic)
                        self.g_stats.append(g_loss.data)
                        self.d_stats.append(d_loss.data)



                    if self.training:
                        if self.cur_epoch != kwargs["ectr"]:
                            self.cur_epoch = kwargs["ectr"]
                            plt.plot(self.i_stats, self.g_stats, '-r', label='G_loss')
                            plt.plot(self.i_stats, self.d_stats, '-b', label='D_loss')
                            plt.legend()
                            plt.savefig(os.path.join(kwargs["outdir"],'loss'+str(kwargs["seed"])+'.png'))
                            plt.clf()
                            pickle.dump([self.i_stats, self.g_stats, self.d_stats], open(os.path.join(kwargs["outdir"],'loss'+str(kwargs["seed"])+'.pkl'), 'wb+'))


                if self.loss_imagination == "adv_gp3":

                    Tensor = torch.cuda.FloatTensor
                    cond = hs.mean(0)
                    self.ic += 1


                    def compute_gradient_penalty(D, real_samples, fake_samples, cond):
                        """Calculates the gradient penalty loss for WGAN GP"""
                        # Random weight term for interpolation between real and fake samples
                        alpha = Tensor(np.random.random((real_samples.size(0), 1)))
                        # Get random interpolation between real and fake samples
                        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
                        interpolates = Variable(torch.cat((interpolates,cond), dim=-1), requires_grad=True)
                        d_interpolates = D(interpolates)
                        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
                        # Get gradient w.r.t. interpolates
                        gradients = autograd.grad(
                            outputs=d_interpolates,
                            inputs=interpolates,
                            grad_outputs=fake,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        return gradient_penalty

                    if (self.cur_d_loss < self.cur_g_loss)[0]:
                        self.critic = 1
                    else:
                        self.critic = 5

                    for _ in range(self.critic):

                        cond_D = cond.detach()

                        if self.training:
                            self.optimizer_D.zero_grad()

                        # Generate a batch of images
                        z = Variable(Tensor(np.random.normal(0, 1, (cond.shape[0], 256))))
                        g_input = torch.cat((cond_D, z), dim=-1)
                        fake_imgs = self.generator(g_input).detach()

                        # Real images
                        d_input1 = torch.cat((image, cond_D), dim=-1)
                        d_input2 = torch.cat((fake_imgs, cond_D), dim=-1)
                        real_validity = self.discriminator(d_input1)
                        fake_validity = self.discriminator(d_input2)

                        # Gradient penalty
                        gradient_penalty = 0
                        if self.training:
                            gradient_penalty = compute_gradient_penalty(self.discriminator, image.data, fake_imgs.data, cond.data)
                        # Adversarial loss
                        d_loss = gamma * (-torch.mean(real_validity) + torch.mean(fake_validity) + self.gradient_penality * gradient_penalty)
                        self.cur_d_loss = d_loss.data

                        if self.training:
                            d_loss.backward(retain_graph=True)
                            self.optimizer_D.step()


                    # Sample noise as generator input
                    self.generator.zero_grad()
                    g_input = torch.cat((cond, z), dim=-1)
                    fake_imgs = self.generator(g_input)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    d_input2 = torch.cat((fake_imgs, cond), dim=-1)
                    fake_validity = self.discriminator(d_input2)
                    g_loss = gamma * (-torch.mean(fake_validity))
                    loss += g_loss


                    #stats
                    self.cur_g_loss = g_loss.data
                    if self.ic % 30==0:
                        print("G_loss {:.3f} D_loss {:.3f}".format(float(g_loss.data), float(d_loss.data)))
                        self.i_stats.append(self.ic)
                        self.g_stats.append(g_loss.data)
                        self.d_stats.append(d_loss.data)



                    if self.training:
                        if self.cur_epoch != kwargs["ectr"]:
                            self.cur_epoch = kwargs["ectr"]
                            plt.plot(self.i_stats, self.g_stats, '-r', label='G_loss')
                            plt.plot(self.i_stats, self.d_stats, '-b', label='D_loss')
                            plt.legend()
                            plt.savefig(os.path.join(kwargs["outdir"],'loss'+str(kwargs["seed"])+'.png'))
                            plt.clf()
                            pickle.dump([self.i_stats, self.g_stats, self.d_stats], open(os.path.join(kwargs["outdir"],'loss'+str(kwargs["seed"])+'.pkl'), 'wb+'))

                #
                if self.loss_imagination == "waae":
                    def compute_gradient_penalty(D, real_samples, fake_samples):
                        """Calculates the gradient penalty loss for WGAN GP"""
                        # Random weight term for interpolation between real and fake samples
                        alpha = Tensor(np.random.random((real_samples.size(0), 1)))
                        # Get random interpolation between real and fake samples
                        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
                        interpolates = Variable(interpolates, requires_grad=True)
                        d_interpolates = D(interpolates)
                        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
                        # Get gradient w.r.t. interpolates
                        gradients = autograd.grad(
                            outputs=d_interpolates,
                            inputs=interpolates,
                            grad_outputs=fake,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                        return gradient_penalty

                    Tensor = torch.cuda.FloatTensor
                    z_sample = hs.mean(0)
                    self.ic += 1
                    self.discriminator.zero_grad()

                    #disc
                    # for _ in range(self.critic):

                    z = Variable(Tensor(np.random.normal(0, 1, (z_sample.shape[0], self.hidden_size))), requires_grad=False)

                    D_real = self.discriminator(z)
                    D_fake = self.discriminator(z_sample.detach())

                    gradient_penalty = compute_gradient_penalty(self.discriminator, z_sample.data, z.data)
                    D_loss = -torch.mean(D_real) + torch.mean(D_fake) + self.gradient_penality * gradient_penalty

                    if self.training:
                        self.optimizer_D.zero_grad()
                        D_loss.backward(retain_graph=True)
                        self.optimizer_D.step()

                    # enc-dec part
                    v = self.img_pool(z_sample)
                    # r_loss = (
                    #     (1 - (torch.acos(F.cosine_similarity(v, image)) / math.pi)).sum()
                    # )
                    r_loss = (
                        F.mse_loss(v,image)
                    )
                    loss += 0.5 * r_loss

                    #dec
                    G_loss = -torch.mean(D_fake)
                    g_loss = 1 * G_loss
                    loss += g_loss

                    #stats

                    if self.ic % 30==0:
                        print("G_loss {:.3f} D_loss {:.3f} R_rec {:.3f}".format(float(G_loss.data), float(D_loss.data), float(r_loss.data)))

                    if self.ic % self.critic*30 == 0:
                        self.i_stats.append(self.ic)
                        self.g_stats.append(G_loss.data)
                        self.d_stats.append(D_loss.data)
                        self.r_stats.append(r_loss.data)

                    if self.training:
                        if self.cur_epoch != kwargs["ectr"]:
                            self.cur_epoch = kwargs["ectr"]
                            plt.plot(self.i_stats, self.g_stats, '-r', label='G_loss')
                            plt.plot(self.i_stats, self.d_stats, '-b', label='D_loss')
                            plt.legend()
                            plt.savefig(os.path.join(kwargs["outdir"],'adv_loss'+str(kwargs["seed"])+'.png'))
                            plt.clf()
                            plt.plot(self.i_stats, self.r_stats, '-r', label='r_loss')
                            plt.legend()
                            plt.savefig(os.path.join(kwargs["outdir"],'r_loss'+str(kwargs["seed"])+'.png'))
                            plt.clf()
                            pickle.dump([self.i_stats, self.g_stats, self.d_stats, self.r_stats], open(os.path.join(kwargs["outdir"],'loss'+str(kwargs["seed"])+'.pkl'), 'wb+'))



        return {'loss': loss, 'logps': logps}

