from functools import partial

import torch
from torch import nn


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


act_dict = {
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "lrelu_03": partial(nn.LeakyReLU, 0.3),
    "relu": nn.ReLU,
    "swish": SWISH,
    None: nn.Identity,
}
