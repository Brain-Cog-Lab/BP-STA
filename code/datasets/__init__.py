__all__ = ['get_dvsc10_data', 'get_dvsg_data', 'get_nmnist_data']

from .cifar10_dvs import CIFAR10DVS
from .dvs128_gesture import DVS128Gesture
from .n_mnist import NMNIST

import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.utils
from torchvision import transforms

'''
https://github.com/fangwei123456/spikingjelly/tree/master/spikingjelly/datasets
'''

DATA_DIR = '/data/datasets'


def get_dvsg_data(batch_size, step):
    train_transform = transforms.Compose([lambda x: torch.tensor(x),
                                          transforms.RandomCrop(128, padding=16),
                                          # transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(5)])
    train_datasets = DVS128Gesture(os.path.join(DATA_DIR, 'DVS/DVS_Gesture'), train=True, transform=train_transform,
                                   data_type='frame', split_by='number', frames_number=step)
    test_datasets = DVS128Gesture(os.path.join(DATA_DIR, 'DVS/DVS_Gesture'), train=False,
                                  data_type='frame', split_by='number', frames_number=step)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2
    )

    return train_loader, test_loader, None, None


def get_dvsc10_data(batch_size, step):
    train_transform = transforms.Compose([lambda x: torch.tensor(x),
                                          transforms.RandomCrop(128, padding=16),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15)])

    train_datasets = CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=train_transform,
                                data_type='frame', split_by='number', frames_number=step)

    test_datasets = CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'),
                               data_type='frame', split_by='number', frames_number=step)

    num_train = len(train_datasets)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    portion = .9
    for i in range(10):
        indices_train.extend(list(range(i * num_per_cls, int(i * num_per_cls + num_per_cls * portion))))
        indices_test.extend(list(range(int(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, drop_last=False, num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, test_loader, None, None


def get_nmnist_data(batch_size, step):
    train_transform = transforms.Compose([lambda x: torch.tensor(x),
                                          transforms.RandomCrop(34, padding=4),
                                          transforms.RandomRotation(10)])
    train_datasets = NMNIST(os.path.join(DATA_DIR, 'DVS/NMNIST'), train=True,
                            data_type='frame', split_by='number', frames_number=step)
    test_datasets = NMNIST(os.path.join(DATA_DIR, 'DVS/NMNIST'), train=False,
                           data_type='frame', split_by='number', frames_number=step)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2
    )

    return train_loader, test_loader, None, None

