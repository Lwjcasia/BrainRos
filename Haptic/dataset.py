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
import pandas as pd
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        sequence_tensor = torch.tensor(sequence, dtype=torch.float).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sequence_tensor, label_tensor


def replace_zeros_with_class_mean(dataframe):
    # 拷贝DataFrame以避免直接修改原始数据
    df = dataframe.copy()
    # 按类别分组并计算每个类别的非零均值
    class_means = df[df.iloc[:, 0] != 0].groupby(df.iloc[:, 1]).mean()
    
    # 遍历每个类别
    for label in class_means.index:
        # 计算当前类别的非零均值
        mean_value = class_means.loc[label].values[0]
        
        # 选择当前类别且值为0的行
        is_class_and_zero = (df.iloc[:, 1] == label) & (df.iloc[:, 0] == 0)
        
        # 替换这些行的值为非零均值
        df.loc[is_class_and_zero, df.columns[0]] = mean_value
    
    return df

def split_sequences_by_label_mid_test(dataframe, sequence_length=16, stride=8, test_ratio=0.2):
    grouped = dataframe.groupby(dataframe.iloc[:, 1])
    train_sequences = []
    train_labels = []
    test_sequences = []
    test_labels = []

    for _, group in grouped:
        features = group.iloc[:, 0].values
        label = group.iloc[:, 1].values[0]

        # 计算中间段测试数据的索引范围
        mid_start_index = int(len(features) * (0.5 - test_ratio / 2))
        mid_end_index = int(len(features) * (0.5 + test_ratio / 2))
        
        # 抽取测试数据
        test_features = features[mid_start_index:mid_end_index]
        test_seq = [test_features[i:i+sequence_length] for i in range(0, len(test_features) - sequence_length + 1, stride)]
        test_sequences.extend(test_seq)
        test_labels.extend([label] * len(test_seq))
        
        # 抽取训练数据（前半部分）
        train_features_front = features[:mid_start_index]
        train_seq_front = [train_features_front[i:i+sequence_length] for i in range(0, len(train_features_front) - sequence_length + 1, stride)]
        train_sequences.extend(train_seq_front)
        train_labels.extend([label] * len(train_seq_front))
        
        # 抽取训练数据（后半部分）
        train_features_back = features[mid_end_index:]
        train_seq_back = [train_features_back[i:i+sequence_length] for i in range(0, len(train_features_back) - sequence_length + 1, stride)]
        train_sequences.extend(train_seq_back)
        train_labels.extend([label] * len(train_seq_back))

    return train_sequences, train_labels, test_sequences, test_labels