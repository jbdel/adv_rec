import torch
import torch.nn as nn
from torch.autograd import Variable

def dropout_mask(x, size, p):

    return Variable(
        torch.bernoulli(
        torch.ones_like(x.new(*size)).mul(1-p)
        ).div(1-p)
    )



class VariationalDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return x * m

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'probability=' + str(self.p) + ')'