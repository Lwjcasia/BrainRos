import torch
import torch.nn as nn
import torch.nn.functional as F  # 使用functional中的ReLu激活函数


class CNN_cifar10(torch.nn.Module):
    def __init__(self):
        super(CNN_cifar10, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # 2为分组大小2*2

    def forward(self, x):
        # 卷积层->池化层->激活函数
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = self.pooling(F.relu(self.conv3(x)))
        x = self.pooling(F.relu(self.conv4(x)))
        x = torch.sigmoid(self.pooling(self.conv5(x)))
        return x


class CNN_MNIST(torch.nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # 2为分组大小2*2

    def forward(self, x):
        # 卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.bn1(self.conv1(x))))
        x = F.relu(self.pooling(self.bn2(self.conv2(x))))
        x = F.relu(self.pooling(self.bn3(self.conv3(x))))
        x = F.relu(self.pooling(self.bn4(self.conv4(x))))
        return x


class CNN_STL10(torch.nn.Module):
    def __init__(self):
        super(CNN_STL10, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        # 池化层
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # 2为分组大小2*2

    def forward(self, x):
        # 卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.bn1(self.conv1(x))))
        x = F.relu(self.pooling(self.bn2(self.conv2(x))))
        x = F.relu(self.pooling(self.bn3(self.conv3(x))))
        x = F.relu(self.pooling(self.bn4(self.conv4(x))))
        return x