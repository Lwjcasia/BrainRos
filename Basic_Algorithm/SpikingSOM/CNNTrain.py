import logging
import torch.nn as nn
import os
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用functional中的ReLu激活函数
import torch.optim as optim
from CNN_sparse import CNN_FC, CNN
from PiCCL import ModelAddProjector, PiCLoss, MNISTstacks

mode='SSL' # 'SSL' or 'SL'

os.environ['NUMEXPR_MAX_THREADS'] = '16'
# Define a transform
# 数据的准备
batch_size = 64
# 神经网络希望输入的数值较小，最好在0-1之间，所以需要先将原始图像(0-255的灰度值)转化为图像张量（值为0-1）
# 仅有灰度值->单通道   RGB -> 三通道 读入的图像张量一般为W*H*C (宽、高、通道数) 在pytorch中要转化为C*W*H

data_path = "./Data/MNIST"

if mode == 'SL':
    model = CNN_FC()
    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root=data_path,train=True,download=True,transform=transform)
    test_dataset = MNIST(root=data_path,train=False,download=True,transform=transform)
elif mode == 'SSL':
    P=4
    model = ModelAddProjector(CNN(), 128,2048,128)
    criterion = PiCLoss(P,1,2).forward
    transform = transforms.Compose([transforms.RandomAffine(degrees=30,translate=(0.3,0.3),scale=(0.8,1.2)),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNISTstacks(P,root=data_path,train=True,download=True,transform=transform)
    test_dataset = MNIST(root=data_path,train=False,download=True,transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size
                          )
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size
                         )

# 在这里加入两行代码，将数据送入GPU中计算！！！
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型的所有内容放入cuda中

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#logging.basicConfig(filename='D:/ChenFei/SPNET/pythonProject/STDP_SOM_program/VGG_ICA/log/CNN128_FC_0104_MSE.log',level=logging.INFO)

# 训练
# 将一次迭代封装入函数中
running_loss = 0.0
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # 在这里data返回输入:inputs、输出target
        inputs, target = data
        # 在这里加入一行代码，将数据送入GPU中计算！！！
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        # 前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for data in test_loader:  # 遍历数据集中的每一个batch
            images, labels = data  # 保存测试的输入和输出
            # 在这里加入一行代码将数据送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 得到预测输出
            _, predicted = torch.max(outputs.data, dim=1)  # dim=1沿着索引为1的维度(行)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))

for epoch in range(10):
    train(epoch)
    #test()
print("Model's state_dict:")

#%%
torch.save(model.state_dict(), './results/CNN/model')


