import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, repeat

from common import ActFun


class AutoEncoder(nn.Module):
    def __init__(self, step, spike_output=True):
        super(AutoEncoder, self).__init__()
        self.step = step
        self.spike_output = spike_output

        # self.gru = nn.GRU(input_size=1, hidden_size=1, num_layers=3)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(1, self.step)
        self.fc2 = nn.Linear(self.step, self.step)
        self.relu = nn.ReLU()
        #
        self.act_fun = ActFun.apply

    def forward(self, x):
        shape = x.shape

        x = self.fc1(x.view(-1, 1))
        x = self.relu(x)
        x = self.fc2(x).transpose_(1, 0)

        # x = x.view(1, -1, 1).repeat(self.step, 1, 1)
        # x, _ = self.gru(x)

        x = self.sigmoid(x)
        if not self.spike_output:
            return x.view(self.step, *shape)
        else:
            return self.act_fun(x).view(self.step, *shape)


# class TransEncoder(nn.Module):
#     def __init__(self, step):
#         super(TransEncoder, self).__init__()
#         self.step = step
#         self.trans = Transformer(dim=128, depth=3, heads=8, dim_head=, mlp_dim, dropout=0.)


class Encoder(nn.Module):
    '''
    (step, batch_size, )
    '''
    def __init__(self, step, encode_type='ttfs'):
        super(Encoder, self).__init__()
        self.step = step
        self.fun = getattr(self, encode_type)
        self.encode_type = encode_type
        if encode_type == 'auto':
            self.fun = AutoEncoder(self.step, spike_output=False)

    def forward(self, inputs, deletion_prob=None, shift_var=None):
        if self.encode_type == 'auto':
            if self.fun.device != inputs.device:
                self.fun.to(inputs.device)

        outputs = self.fun(inputs)
        if deletion_prob:
            outputs = self.delete(outputs, deletion_prob)
        if shift_var:
            outputs = self.shift(outputs, shift_var)
        return outputs

    @torch.no_grad()
    def direct(self, inputs):
        shape = inputs.shape
        outputs = inputs.unsqueeze(0).repeat(self.step, *([1] * len(shape)))
        return outputs

    def auto(self, inputs):
        shape = inputs.shape
        outputs = self.fun(inputs)
        print(outputs.shape)
        return outputs

    @torch.no_grad()
    def ttfs(self, inputs):
        # print("ttfs")
        shape = (self.step,) + inputs.shape
        outputs = torch.zeros(shape, device=self.device)
        for i in range(self.step):
            mask = (inputs * self.step <= (self.step - i)) & (inputs * self.step > (self.step - i - 1))
            outputs[i, mask] = 1 / (i + 1)
        return outputs

    @torch.no_grad()
    def rate(self, inputs):
        shape = (self.step,) + inputs.shape
        return (inputs > torch.rand(shape, device=self.device)).float()

    @torch.no_grad()
    def phase(self, inputs):
        shape = (self.step,) + inputs.shape
        outputs = torch.zeros(shape, device=self.device)
        inputs = (inputs * 256).long()
        val = 1.
        for i in range(self.step):
            if i < 8:
                mask = (inputs >> (8 - i - 1)) & 1 != 0
                outputs[i, mask] = val
                val /= 2.
            else:
                outputs[i] = outputs[i % 8]
        return outputs

    @torch.no_grad()
    def delete(self, inputs, prob):
        mask = (inputs >= 0) & (torch.randn_like(inputs, device=self.device) < prob)
        inputs[mask] = 0.
        return inputs

    @torch.no_grad()
    def shift(self, inputs, var):
        outputs = torch.zeros_like(inputs)
        for step in range(self.step):
            shift = (var * torch.randn(1)).round_() + step
            shift.clamp_(min=0, max=self.step - 1)
            outputs[step] += inputs[int(shift)]
        return outputs
