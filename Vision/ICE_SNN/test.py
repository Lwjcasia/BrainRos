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
from model import spiking_resnet18_ICE
from dataset import cifar10_dataset

_seed_ = 666
torch.manual_seed(_seed_)
np.random.seed(_seed_)
random.seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 20231229 在原来的基础上，加了CIFAR10Policy、cutout，训练400个epoch，删除验证集
# 设定超参数
num_classes = 10  # 分类数目
batch_size = 64  # 批量大小
num_epochs = 400  # 训练周期
T = 4  # 时间步数

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = cifar10_dataset(batch_size=batch_size)
# 实例化网络
model = spiking_resnet18_ICE(T=T, num_classes=num_classes).to(device)
check_point = torch.load('your/path/')['model_state']
model.load_state_dict(check_point)
print(f'time step = {T}')
loss_fun = nn.CrossEntropyLoss()
# 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)

restart = False 
if restart:
    state = torch.load('model_checkpoint_ybh2.pth')
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    start_epoch = state['epoch']   
    scheduler = state['scheduler']
else:
    start_epoch = 0 
best_accuracy = 0.0  # 用于跟踪最佳验证准确率
best_model_state = None  # 用于保存最佳模型状态


model.eval()
with torch.no_grad():           
# 测试循环
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        functional.reset_net(model)  # 重置网络状态
        output = model(data)
        test_loss += loss_fun(output, target).item()  # 累加损失
        pred = output.data.max(1, keepdim=True)[1]  # 获取最大概率的索引
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_accuracy))

