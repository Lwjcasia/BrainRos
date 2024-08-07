import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
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
from sklearn.model_selection import train_test_split
import time

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.iloc[:, :-1].values  # 假设前面的列都是特征
        self.labels = dataframe.iloc[:, -1].values  # 假设最后一列是标签
        self.num_samples = len(self.data) // 16

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 将16行合并成一个样本
        start_idx = idx * 16
        end_idx = start_idx + 16
        # print(self.data[start_idx:end_idx])
        # print(self.labels[start_idx:start_idx+1])
        sample_features = self.data[start_idx:end_idx].reshape(4, 16)

        sample_label = self.labels[start_idx:start_idx+1].reshape(-1)  # 根据实际情况调整标签的形状
        return torch.tensor(sample_features, dtype=torch.float).unsqueeze(0), torch.tensor(sample_label, dtype=torch.long).squeeze()