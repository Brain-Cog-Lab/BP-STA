# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingBCEWithLogitsLoss(nn.Module):

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        target = torch.eye(x.shape[-1], device=x.device)[target]
        nll = torch.ones_like(x) / x.shape[-1]
        return self.BCELoss(x, target) * self.confidence + self.BCELoss(x, nll) * self.smoothing


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()


class SoftCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, inputs, targets, temperature=1.):
        log_likelihood = -F.log_softmax(inputs / temperature, dim=1)
        likelihood = F.softmax(targets / temperature, dim=1)
        sample_num, class_num = targets.shape
        loss = torch.sum(torch.mul(log_likelihood, likelihood)) / sample_num
        return loss


class UnilateralMse(torch.nn.Module):
    def __init__(self, thresh):
        super(UnilateralMse, self).__init__()
        self.thresh = thresh
        self.loss = torch.nn.MSELoss()

    def forward(self, x, target):
        # x = nn.functional.softmax(x, dim=1)
        torch.clip(x, max=self.thresh)
        # print(x)
        return self.loss(x, torch.zeros(x.shape, device=x.device).scatter_(1, target.view(-1, 1), self.thresh))


class WarmUpLoss(torch.nn.Module):
    def __init__(self):
        super(WarmUpLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, x, target, epoch=15):
        x = nn.functional.softmax(x, dim=1)
        return self.ce(x, target)



