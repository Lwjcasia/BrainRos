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
from dataset import replace_zeros_with_class_mean, CustomDataset, split_sequences_by_label_mid_test
from model import spiking_resnet18
_seed_ = 666
torch.manual_seed(_seed_)
np.random.seed(_seed_)
random.seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    
num_classes = 2  # 分类数目
learning_rate = 1e-3  # 学习率
batch_size = 64  # 批量大小
num_epochs = 300  # 训练周期

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用示例
df = pd.read_excel('yali/total.xlsx', header=None)  # 加载数据
df = replace_zeros_with_class_mean(df)  
sequence_length = 16
train_sequences, train_labels, test_sequences, test_labels = split_sequences_by_label_mid_test(df, sequence_length=sequence_length, stride=1)

# 创建训练集和测试集的CustomDataset实例
train_dataset = CustomDataset(train_sequences, train_labels)
test_dataset = CustomDataset(test_sequences, test_labels)

# 接下来，可以像平常一样使用DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 划分数据集为训练集和测试集

# 实例化网络
model = spiking_resnet18(T=4).to(device)
print(model)
loss_fun = nn.CrossEntropyLoss()
# 定义优化器
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cos_lr_T)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=75,gamma=0.5)
a = []
ac_list = []


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
with open('results_haptic_1dspikingresnet18.txt', 'w') as file:
    for epoch in range(start_epoch, num_epochs):
        
        model.train()  # 将模型设置为训练模式
        t1 = time.time()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            #print(i)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            functional.reset_net(model)
            output = model(data)
            # print(target.shape)
            # print(output.shape)
            loss = loss_fun(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        scheduler.step()
        avg_loss = train_loss / total
        accuracy = 100. * correct / total
        file.write(f"Epoch {epoch+1}: Average Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Epoch {epoch+1}: Average Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        t2 = time.time()
        file.write(f"训练时间为: {t2-t1}")
        print("训练时间为",t2-t1)
    # 关闭梯度计算

        model.eval()

        # 测试循环
        with torch.no_grad():   
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
            t3 = time.time()
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f},time:{:.4f}\n'.format(test_loss, test_accuracy, t3-t2))
            file.write(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Time: {t3-t2}\n')
            file.flush()
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }
            torch.save(best_model_state, f'best_resnet18_1dsnnresnet18_haptic.pth')
        print('bestacc{:.4f}\n'.format(best_accuracy))
        file.write('bestacc{:.4f}\n'.format(best_accuracy))
