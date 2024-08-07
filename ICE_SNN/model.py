import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer




class Basicblockann(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblockann, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3,padding=
                          1,  stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        out1 = self.shortcut(x) + out1
        out1 = F.relu(out1)
        out2 = self.conv3(out1)
        out2 = self.conv4(out2)
        out2 = out1 + out2 
        # out = F.relu(out)
        return out2
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('MemAddBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in MemAddBasicBlock")


        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.sn1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')

        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False),
            norm_layer(planes)
        )
        self.sn2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')

        self.downsample = downsample
        
        self.conv3 = layer.SeqToANNContainer(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.sn3 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')

        self.conv4 = layer.SeqToANNContainer(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False),
            norm_layer(planes)
        )
    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.sn2(out)
        out2 = self.conv3(out)
        out2 = self.sn3(out2)
        out2 = self.conv4(out2)
        out2 += out
        return out2

def zero_init_blocks(net: nn.Module):
    for m in net.modules():
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)

class SpikingResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4):
        super(SpikingResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.annconv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.annbn1 = nn.BatchNorm2d(self.inplanes)
        self.sn1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')
        #self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                      dilate=replace_stride_with_dilation[0])

        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.lif4 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), step_mode='m')
        
        self.flatten = layer.SeqToANNContainer(nn.Flatten())
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512, num_classes)
        #self.snsn = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')
        self.annlayer1 = Basicblockann(64, 64)
        self.annlayer2 = Basicblockann(64, 128, stride=2)
        self.annlayer3 = Basicblockann(128, 256, stride=2)


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
        if stride != 1 or self.inplanes != planes:
            downsample = layer.SeqToANNContainer(
                    nn.Conv2d(self.inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False),
                    norm_layer(planes),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer
                            ))
        self.inplanes = planes 
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer
                                ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        annx = self.annconv1(x)
        annx = self.annbn1(annx)
        annx = F.relu(annx)
        #annx = self.maxpool(annx)
        annx1 = self.annlayer1(annx)
        annx1_relu = F.relu(annx1)
        annx1_add = annx1_relu.unsqueeze(0)
        annx1_add = annx1_add.repeat(self.T, 1, 1, 1, 1)

        annx2 = self.annlayer2(annx1_relu)
        annx2_relu = F.relu(annx2)
        annx2_add = annx2_relu.unsqueeze(0)
        annx2_add = annx2_add.repeat(self.T, 1, 1, 1, 1)

        annx3 = self.annlayer3(annx2_relu)
        annx3_relu = F.relu(annx3)
        annx3_add = annx3_relu.unsqueeze(0)
        annx3_add = annx3_add.repeat(self.T, 1, 1, 1, 1)

        snnx = self.conv1(x)
        snnx = self.bn1(snnx)
        snnx.unsqueeze_(0)
        snnx = snnx.repeat(self.T, 1, 1, 1, 1)
        out = self.sn1(snnx)
        #print(out.shape)
        #out = self.maxpool(out)
        # x = self.maxpool(x)
        out = self.layer1(out)
        out += annx1_add
        out = self.lif1(out)
        out = self.layer2(out)
        out += annx2_add
        out = self.lif2(out)
        out = self.layer3(out)
        out += annx3_add
        out = self.lif3(out)
        out = self.layer4(out)
        out = self.lif4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out.mean(0))
        return out

    def forward(self, x):
        return self._forward_impl(x)
def _spiking_resnet(block, layers, **kwargs):
    model = SpikingResNet(block, layers, **kwargs)
    return model


def spiking_resnet18_ICE(**kwargs):

    return _spiking_resnet(BasicBlock, [1, 1, 1, 1], **kwargs)