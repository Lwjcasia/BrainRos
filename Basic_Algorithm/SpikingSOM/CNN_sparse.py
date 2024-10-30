import torch
import torch.nn as nn
import torch.nn.functional as F  # 使用functional中的ReLu激活函数


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # 2为分组大小2*2

    def forward(self, x):
        batch_size = x.size(0)
        # 卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = F.relu(self.pooling(self.conv4(x)))
        x = x.view(batch_size, -1)
        x = F.normalize(x,dim=1)
        return x


class CNN_FC(torch.nn.Module):
    def __init__(self):
        super(CNN_FC, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # 2为分组大小2*2
        # 全连接层 1152 = 128 * 3 * 3
        self.fc = nn.Linear(128, 10,)

    def forward(self, x):
        # 先从x数据维度中得到batch_size
        batch_size = x.size(0)
        # 卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = F.relu(self.pooling(self.conv4(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
