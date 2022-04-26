import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from common import ActFun


class DGLIFNode(nn.Module):
    def __init__(self,
                 threshold=.5,
                 shape=None,
                 decay=1.):
        super(DGLIFNode, self).__init__()
        self.shape = shape

        self.mem = None
        self.spike = None
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        self.decay = Parameter(torch.tensor(decay), requires_grad=False)
        self.n_reset()

    def n_reset(self):
        self.mem = None #torch.zeros(self.shape, device=self.device)
        self.spike = None #torch.zeros(self.shape, device=self.device)

    def integral(self, inputs):
        if self.mem is None:
            self.mem = inputs
        else:
            self.mem += inputs

    def calc_spike(self):
        # thresh = nn.Sigmoid()(self.threshold)
        # self.spike = self.mem.clone() / thresh
        # self.spike[self.spike < 1.] = 0.
        # self.spike = thresh.detach() * self.spike / (self.mem.detach().clone() + 1e-12)
        # self.mem[self.mem >= thresh] = 0.
        if True: #self.training:
            self.spike = self.mem.clone()
            self.spike[(self.spike < self.threshold)] = 0.
            self.spike = self.spike / (self.mem.detach().clone() + 1e-12) #* self.threshold
            # print(self.spike)
            self.mem[(self.mem >= self.threshold)] = 0.

        self.mem = self.mem * self.decay

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike

    def get_fire_rate(self):
        return float((self.spike.detach().sum())) / float(np.product(self.spike.shape))

    def get_mem_loss(self):
        spike = self.spike[self.spike > 0.]
        return (spike - 1.) ** 2


class HTGLIFNode(nn.Module):
    def __init__(self,
                 threshold=.5,
                 shape=None,
                 decay=1.):
        super(HTGLIFNode, self).__init__()
        self.shape = shape

        self.mem = None
        self.spike = None
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        self.decay = Parameter(torch.tensor(decay), requires_grad=False)

        self.warm_up = False

        self.n_reset()

    def n_reset(self):
        self.mem = None # torch.zeros(self.shape, device=self.device)
        self.spike = None # torch.zeros(self.shape, device=self.device)

    def integral(self, inputs):
        if self.mem is None:
            self.mem = inputs
        else:
            self.mem += inputs

    def calc_spike(self):
        spike = self.mem.clone()
        spike[(spike < self.threshold)] = 0.
        self.spike = spike / (self.mem.detach().clone() + 1e-12)
        self.mem = torch.where(self.mem >= self.threshold, torch.zeros_like(self.mem), self.mem)

        self.mem = self.mem + 0.2 * self.spike - 0.2 * self.spike.detach()
        self.mem = self.mem * self.decay

    def forward(self, inputs):
        if self.warm_up:
            return inputs
        else:
            self.integral(inputs)
            self.calc_spike()
            return self.spike

    def get_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(np.product(self.spike.shape))

    def get_mem_loss(self):
        spike = self.spike[self.spike > 0.]
        return (spike - 1.) ** 2

    def set_n_warm_up(self, flag):
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)


class SGLIFNode(nn.Module):
    def __init__(self,
                 shape=None,
                 threshold=None,
                 decay=1.):
        super(SGLIFNode, self).__init__()
        self.shape = shape
        self.act_fun = ActFun.apply

        self.decay = Parameter(torch.tensor(decay), requires_grad=False)
        self.mem = None
        self.spike = None
        self.n_reset()

    def n_reset(self):
        self.mem = None
        self.spike = None

    def integral(self, inputs):
        if self.mem is None:
            self.mem = inputs
        else:
            self.mem += inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem)
        self.mem = self.mem * self.decay * (1. - self.spike)
        # self.spike = self.mem.clone()
        # self.spike[(self.spike < 1.) & (self.spike > -1.)] = 0.
        # self.mem = self.mem * self.decay
        # self.mem[(self.mem >= 1.) | (self.mem <= -1.)] = 0.
        return self.spike

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike

    def get_fire_rate(self):
        return float((self.spike.detach().sum())) / float(np.product(self.spike.shape))
