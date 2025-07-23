import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer



decay = 0.25  # 0.25 # decay constants
class MultiSpike4(nn.Module):  # 直接调用实例化的quant6无法实现深拷贝。解决方案是像下面这样用嵌套的类  #注意135的神经元就是用的这个，不是mem

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (torch.round(torch.clamp(input*100, min=0, max=511)))/100

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 5.11] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)
    
class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = MultiSpike4()  #修改时只要修改这里就好  111111

    def forward(self, x):
#         print("=================")

        mem = torch.zeros_like(x[0]).to(x.device)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                # mem = mem_old * decay * (1 - spike.detach()) + x[i]
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])

        return output
    
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
        self.sn1 = mem_update()

        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False),
            norm_layer(planes)
        )
        self.sn2 = mem_update()
        self.downsample = downsample
        # self.stride = stride
        # self.conv3 = layer.SeqToANNContainer(
        #     nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.BatchNorm2d(planes)
        # )
        # self.sn3 = mem_update()

        # self.conv4 = layer.SeqToANNContainer(
        #     nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False),
        #     norm_layer(planes)
        # )
        # self.sn4 = mem_update()
    def forward(self, x):
        
        identity = x
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn2(out)
        # out2 = self.conv3(out)
        # out2 = self.sn3(out2)
        # out2 = self.conv4(out2)
        # out2 += out
        return out
def zero_init_blocks(net: nn.Module):
    for m in net.modules():
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)


class SpikingResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=1):
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)


        self.sn1 = mem_update()
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(3,2,1))
    
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.flatten = layer.SeqToANNContainer(nn.Flatten())
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.snsn = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')

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
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3, padding=1, stride=stride),
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
        x = x.repeat(self.T, 1, 1, 1, 1)
        out = self.sn1(x)
        
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out.mean(0))
        return out

    def forward(self, x):
        return self._forward_impl(x)
def _spiking_resnet(block, layers, **kwargs):
    model = SpikingResNet(block, layers, **kwargs)
    return model


def spiking_resnet18(**kwargs):

    return _spiking_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def spiking_resnet34(**kwargs):

    return _spiking_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)