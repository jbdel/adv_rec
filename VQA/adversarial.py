import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


def compute_gradient_penalty(D, real_samples, fake_samples, cond=None, mode="wgan"):

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor
    if mode == "wgan":
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    else:
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


def run_wgan(G, D, optimizer_D, penality, critic, h, v):

    Tensor = torch.cuda.FloatTensor
    bs = h.size(0)

    #confidional is last hidden state
    cond = h

    #tiling h to match v dimension
    cond_tiled = cond.view(bs, cond.size(1), 1, 1)
    tiled_dims = [-1, -1, 6, 6]
    cond_tiled = cond_tiled.expand(*tiled_dims)
    #reshaping v
    v = v.permute(0,2,1).view(bs, -1, 6, 6).contiguous().detach()

    z = Variable(Tensor(np.random.normal(0, 1, (bs, 512))))
    G.zero_grad()
    ret_fake_imgs = G(h)
    g_loss = -torch.mean(D(ret_fake_imgs))

    d_loss = torch.zeros(1)
    for _ in range(critic):

        if(g_loss<1.0):

            optimizer_D.zero_grad()
            D.zero_grad()
            # Generate a batch of images
            z = Variable(Tensor(np.random.normal(0, 1, (bs, 512))))
            # g_input = torch.cat((cond.detach(), z), dim=-1)
            fake_imgs = G(cond.detach())

            # Real images
            d_input1 = v
            d_input2 = fake_imgs
            real_validity = D(d_input1)
            fake_validity = D(d_input2)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(D, v.data, fake_imgs.data, None, mode="wgan")
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + penality * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

    # print(g_loss, d_loss)
    return g_loss, d_loss, ret_fake_imgs, v
#
# def run_waae(G, D, optimizer_D, penality, critic, h, v):
#     import math
#     def log_density_igaussian(z, z_var):
#         """Calculate log density of zero-mean isotropic gaussian distribution given z and z_var."""
#         assert z.ndimension() == 2
#         assert z_var > 0
#
#         z_dim = z.size(1)
#
#         return -(z_dim / 2) * math.log(2 * math.pi * z_var) + z.pow(2).sum(1).div(-2 * z_var)
#
#     Tensor = torch.cuda.FloatTensor
#     z_sample = h
#     bs = h.size(0)
#
#     ones = Variable(torch.ones(z_sample.shape[0], 1)).cuda()
#     zeros = Variable(torch.zeros(z_sample.shape[0], 1)).cuda()
#
#     # z = Variable(Tensor(np.random.normal(0, 1, (z_sample.shape[0], self.num_hid))))
#     z = Variable(math.sqrt(2) * torch.randn(z_sample.shape[0], 1024)).cuda()
#     log_p_z = log_density_igaussian(z, 2).view(-1, 1)
#
#     D_fake = D(z)
#     D_real = D(z_sample.detach())
#     if critic == 5:
#         loss_d = F.binary_cross_entropy_with_logits(D_fake + log_p_z, ones) + \
#                  F.binary_cross_entropy_with_logits(D_real + log_p_z, zeros)
#     else:
#         loss_d = F.binary_cross_entropy_with_logits(D_fake, ones) + \
#                  F.binary_cross_entropy_with_logits(D_real, zeros)
#
#     gradient_penalty = compute_gradient_penalty(D, z_sample.data, z.data, mode="waae")
#
#     D_loss = loss_d + penality * gradient_penalty
#
#     optimizer_D.zero_grad()
#     loss_d.backward()
#     optimizer_D.step()
#
#     # enc-dec part
#     fake_imgs = G(z_sample)
#     if critic == 5:
#         G_loss = F.binary_cross_entropy_with_logits(D_real.detach() + log_p_z, ones)
#     else:
#         G_loss = F.binary_cross_entropy_with_logits(D_real.detach(), ones)
#     v = v.permute(0,2,1).view(bs,-1, 6, 6)
#     return G_loss, D_loss, fake_imgs, v


def run_waae(G, D, optimizer_D, penality, critic, h, v):
    Tensor = torch.cuda.FloatTensor
    bs = h.size(0)
    z_sample = h.detach()

    ones = Variable(torch.ones(z_sample.shape[0], 1)).cuda()
    zeros = Variable(torch.zeros(z_sample.shape[0], 1)).cuda()

    # enc-dec part
    fake_imgs = G(h)
    D_fake = D(h)
    # print(D[0].weight)
    G_loss = F.binary_cross_entropy_with_logits(D_fake, ones)
    v = v.permute(0,2,1).view(bs,-1, 6, 6)

    D_loss = torch.zeros(1)
    for _ in range(critic):

        if(G_loss<1.0):

            optimizer_D.zero_grad()
            D.zero_grad()

            z = Variable(Tensor(np.random.normal(0, 1, (z_sample.shape[0], 1024))), requires_grad=False)

            D_real = D(z)
            D_fake = D(z_sample)

            gradient_penalty = compute_gradient_penalty(D, z_sample.data, z.data, mode="waae")

            D_loss = F.binary_cross_entropy_with_logits(D_fake, zeros) + \
                                 F.binary_cross_entropy_with_logits(D_real, ones)  + penality * gradient_penalty

            D_loss.backward()
            optimizer_D.step()


    # print(G_loss, D_loss)
    return G_loss, D_loss, fake_imgs, v



class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        # self.up_block = nn.Sequential(
        #         ConvBlock(1024+self.noise_dim, 512, kernel_size=2),
        #         nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2),
        #         ConvBlock(512, 256, kernel_size=2),
        #         nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),
        #         ConvBlock(256, 256, kernel_size=2),
        #     )

        # self.up_block = nn.Sequential(
        #         nn.ConvTranspose2d(1024+self.noise_dim, 512, kernel_size=3),
        #     nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            # ConvBlock(512, 256, kernel_size=2),
            # nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            # ConvBlock(256, 512, kernel_size=2),
            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            # ConvBlock(512, 1024, kernel_size=3),
        #     )

        #feat36
        self.up_block = nn.Sequential(
        ConvBlock(1024+self.noise_dim, 512, kernel_size=2),
        nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2),
        ConvBlock(512, 2048, kernel_size=2),


        )



    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1, 1, 1)
        x = self.up_block(x)
        return x




class Discriminator(nn.Module):
    def __init__(self, num_hid=1024):

        super(Discriminator, self).__init__()
        self.num_hid = num_hid

        # basic_block = torchvision.models.resnet.BasicBlock
        # self.layer4 = _make_layer(1024+256, basic_block, 512, 2, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 1)
        #

        # self.layer4 = _make_layer(1024, basic_block, 512, 6, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.down_blocks = nn.Sequential(
        #     ConvBlock(256, 256, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     ConvBlock(256, 128, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     ConvBlock(128, 128, kernel_size=3)
        # )
        # self.fc = nn.Linear(2048, 1)

        #36 feat
        # self.down_blocks = nn.Sequential(
        #
        #     ConvBlock(2048+self.num_hid, 512, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     ConvBlock(512, 256, kernel_size=3),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     ConvBlock(256, 128, kernel_size=3),
        #
        # )
        # self.fc = nn.Linear(512, 1)

        self.input = nn.Sequential(
            nn.Linear(2048 + self.num_hid, 512),
            nn.ReLU(),
        )

        self.down_blocks = nn.Sequential(
            ConvBlock(512, 256, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 128, kernel_size=3),

        )

        self.fc = nn.Linear(128*3*3, 1)


    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(1))
        x = self.input(x)
        x = x.view(x.size(0), x.size(2), 6, 6)
        x = self.down_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
