
import numpy as np
import torch.nn as nn



class Generator_adv(nn.Module):
    def __init__(self, hidden_size):
        super(Generator_adv, self).__init__()
        self.hidden_size = hidden_size

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            # *block(100, 128),
            # *block(128, 256),
            # *block(256, 512),
            *block(512+256, 1024),
            nn.Linear(1024, 2048),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

class Discriminator_adv(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator_adv, self).__init__()
        self.hidden_size = hidden_size


        self.model = nn.Sequential(
            nn.Linear(2048+self.hidden_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity