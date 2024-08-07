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

tau = 2.0
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('MemAddBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in MemAddBasicBlock")


        self.conv1 = layer.SeqToANNContainer(
            nn.Conv1d(inplanes, planes, kernel_size=5, padding=2, stride=stride, bias=False),
            #nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes)
        )
        self.sn1 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')

        self.conv2 = layer.SeqToANNContainer(
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, stride=1, bias=False),
            #nn.Conv1d(planes, planes, kernel_size=5, padding=2, stride=1, bias=False),
            norm_layer(planes)
        )
        self.sn2 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')
        self.downsample = downsample
        self.stride = stride
        self.conv3 = layer.SeqToANNContainer(
            nn.Conv1d(planes, planes, kernel_size=5, padding=2, stride=1, bias=False),
            #nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(planes)
        )
        self.sn3 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')

        self.conv4 = layer.SeqToANNContainer(
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, stride=1, bias=False),
            #nn.Conv1d(planes, planes, kernel_size=5, padding=2, stride=1, bias=False),
            norm_layer(planes)
        )
        self.sn4 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')
    def forward(self, x):
        
        identity = x
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn2(out)
        out2 = self.conv3(out)
        out2 = self.sn3(out2)
        out2 = self.conv4(out2)
        out2 += out
        return self.sn4(out2)
def zero_init_blocks(net: nn.Module):
    for m in net.modules():
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)

class SpikingResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4):
        super(SpikingResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)


        self.sn1 = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')
        #self.maxpool = layer.SeqToANNContainer(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.flatten = layer.SeqToANNContainer(nn.Flatten())
        self.avgpool = layer.AdaptiveAvgPool1d((1, 1))
        #self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool1d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.snsn = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            zero_init_blocks(self)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = layer.SeqToANNContainer(
                    nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=3, padding=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1)
        out = self.sn1(x)
        #out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = out.mean(-1, keepdim=True)
        #out = self.avgpool(out)
        #print(out.shape)
        out = self.flatten(out)
        #print(out.shape)
        out = self.fc(out.mean(0))
        return out

    def forward(self, x):
        return self._forward_impl(x)
def _spiking_resnet(block, layers, **kwargs):
    model = SpikingResNet(block, layers, **kwargs)
    return model


def spiking_resnet18(**kwargs):

    return _spiking_resnet(BasicBlock, [1, 1, 1, 1], **kwargs)