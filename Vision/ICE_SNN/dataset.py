import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
torch.autograd.set_detect_anomaly(True)
import numpy as np
import random
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import time
from autoaugment import CIFAR10Policy, Cutout


def cifar10_dataset(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize((0.48836562, 0.48134598, 0.4451678), (0.24833508, 0.24547848, 0.26617324))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47375134, 0.47303376, 0.42989072), (0.25467148, 0.25240466, 0.26900575))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./CIFAR10',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./CIFAR10',
        train=False,
        download=True,
        transform=transform_test
    )



    # 定义数据加载器
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        worker_init_fn=np.random.seed(666)
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=batch_size
            )
    return train_loader, test_loader


def cifar100_dataset(batch_size):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
    transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_dataset = torchvision.datasets.CIFAR100(
        root='./CIFAR100',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root='./CIFAR100',
        train=False,
        download=True,
        transform=transform_test
    )

    # 定义数据加载器
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=batch_size
            )