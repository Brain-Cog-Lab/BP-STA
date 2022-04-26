import os
import sys

import numpy as np
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from nodes import *


class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class NDropout(nn.Module):
    def __init__(self, p):
        super(NDropout, self).__init__()
        self.p = p
        self.mask = None

    def n_reset(self):
        self.mask = None

    def create_mask(self, x):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return self.mask * x
        else:
            return x
