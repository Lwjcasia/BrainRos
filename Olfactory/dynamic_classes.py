import serial
import numpy as np
import torch
import spikingjelly
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
from model import spiking_resnet
# 初始化串口
ser = serial.Serial('COM3', 9600, timeout=1)

# 初始化数据缓存
data_buffer = []
T = 4  # 时间步数
tau = 2.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 9  # 分类数目
# 初始化网络（替换为您自己的网络加载代码）
# model = load_your_model()

# 实例化网络

model = spiking_resnet(T=4).to(device)
checkpoint = torch.load('model_tiqi_checkpoint64.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
for i in range(20):
    data = ser.readline()
while True:
    # 读取串口数据
    data = ser.readline()

    if data:
        # 解析数据
        # 数据格式为 MQ138: 76  MQ135: 76  MQ8: 35  MQ3: 56
        values = [int(value.split(": ")[1]) for value in data.decode('utf-8').strip().split("  ")]
        data_buffer.append(values)

        # 检查是否收集了足够的数据
        if len(data_buffer) == 16:
            # 转换为 NumPy 数组并重塑为 (1, 16, 4) 用于网络预测
            # Converting the list to a 16x4 tensor
            input_data = np.array(data_buffer).reshape(1, 1, 16, 4)
            input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
            #print(input_data.shape)
            print(input_data)
            # 网络预测
            functional.reset_net(model)
            prediction = model(input_data)
            _, predicted = torch.max(prediction.data, 1)
            # 输出预测结果（或进行其他处理）
    
            if predicted.item() == 0:
                print('正常')
            elif predicted.item() == 1:
                print('乙醇：轻微级')
            elif predicted.item() == 2:
                print('乙醇：警告级')
            elif predicted.item() == 3:
                print('乙醇：危险级')
            elif predicted.item() == 4:
                print('乙醇：极端危险级')
            elif predicted.item() == 5:
                print('丁烷：轻微级')
            elif predicted.item() == 6:
                print('丁烷：警告级')
            elif predicted.item() == 7:
                print('丁烷：危险级')
            elif predicted.item() == 8:
                print('丁烷：极端危险级')
                # 清空数据缓存以进行下一轮收集
            data_buffer = []
