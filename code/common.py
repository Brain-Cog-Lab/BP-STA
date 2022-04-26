import os
import sys

import numpy as np
import torch
from torch import nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torchvision import transforms
import torch.backends.cudnn as cudnn

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return inputs.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        grad_inputs = grad_output.clone()
        temp = abs(inputs - thresh) < lens
        return grad_inputs * temp.float()
        # grad = torch.exp((thresh - inputs)) / ((torch.exp((thresh - inputs)) + 1) ** 2)
        # return grad * grad_output


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Compute the top1 and top5 accuracy

    """
    maxk = max(topk)
    batch_size = target.size(0)

    # Return the k largest elements of the given input tensor
    # along a given dimension -> N * k
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res