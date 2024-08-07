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
from dataset import CustomDataset
from model import spiking_resnet

_seed_ = 666
torch.manual_seed(_seed_)
np.random.seed(_seed_)
random.seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# 设定超参数
num_classes = 9  # 分类数目
learning_rate = 1e-3  # 学习率
batch_size = 64  # 批量大小
num_epochs = 64  # 训练周期
T = 4  # 时间步数
tau = 2.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 假设 df 是您从Excel文件中读取的DataFrame
df = pd.read_excel('data.xlsx', header=None)


group_size = 16
num_groups = len(df) // group_size

df['group'] = [i // group_size for i in range(num_groups * group_size)]

# 基于组划分数据集
grouped_df = df.groupby('group')

train_groups, test_groups = train_test_split(list(grouped_df.groups.keys()), test_size=0.2)

train_df = pd.concat([grouped_df.get_group(g) for g in train_groups])
test_df = pd.concat([grouped_df.get_group(g) for g in test_groups])

train_df = train_df.drop('group', axis=1)

test_df = test_df.drop('group', axis=1)

train_dataset = CustomDataset(train_df)
test_dataset = CustomDataset(test_df)

# 使用DataLoader为训练集和测试集创建加载器
train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False)


# 实例化网络
model = spiking_resnet(T=4).to(device)
checkpoint = torch.load('model_tiqi_checkpoint64.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
#print(model)
loss_fun = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cos_lr_T)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
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
# 初始化每个类别的正确预测数和总数
correct_pred = [0] * 9
total_pred = [0] * 9
with open('results.txt', 'w') as file:
    for epoch in range(start_epoch, num_epochs):
        

        model.train()  # 将模型设置为训练模式
        t1 = time.time()
        train_loss = 0
        correct = 0
        total = 0
        
        for i, (data, target) in enumerate(train_dataloader):
            #print(i)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            functional.reset_net(model)
            output = model(data)
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


        model.eval()
        with torch.no_grad():           
        # 测试循环
            test_loss = 0
            correct = 0
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                functional.reset_net(model)  # 重置网络状态
                output = model(data)
                test_loss += loss_fun(output, target).item()  # 累加损失
                pred = output.data.max(1, keepdim=True)[1]  # 获取最大概率的索引
                correct += pred.eq(target.data.view_as(pred)).sum().item()
                if epoch == num_epochs - 1:
                    for label, prediction in zip(target, pred):
                        if label == prediction:
                            correct_pred[label] += 1
                        total_pred[label] += 1

            test_loss /= len(test_dataloader.dataset)
            test_accuracy = correct / len(test_dataloader.dataset)
            t3 = time.time()
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f},time:{:.4f}\n'.format(test_loss, test_accuracy, t3-t2))
            file.write(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Time: {t3-t2}\n')
            file.flush()
        if epoch == num_epochs - 1:
            torch.save({
                'model_state': model.state_dict(),
            }, 'model_checkpoint.pth')
    for class_idx in range(9):
        print(total_pred)
        print(correct_pred)
        accuracy = 100 * float(correct_pred[class_idx]) / total_pred[class_idx]
        print(f'Accuracy for class {class_idx}: {accuracy:.4f}%')

# 将每个类别的准确率写入文件
    with open('class_accuracy_results.txt', 'w') as f:
        for class_idx in range(9):
            accuracy = 100 * float(correct_pred[class_idx]) / total_pred[class_idx]
            f.write(f'Accuracy for class {class_idx}: {accuracy:.4f}%\n')

            

