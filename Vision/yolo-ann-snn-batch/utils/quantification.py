import torch
import torch.nn as nn

class Quantification(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, timesteps=0):
        assert timesteps != 0, "timesteps must be set to nozero"
        precision = 1/timesteps
        #　通常用于减少模型的大小和计算复杂度
        c = x.shape[1]
        max_channel_value = x.transpose(0,1).reshape(c,-1).max(dim=1)[0].detach()
        x = x.transpose(1,-1)
        x = x.div(max_channel_value)
        x =  (x/precision).int()*precision
        x  = x.mul(max_channel_value)
        x = x.transpose(1,-1)
        return x

    @staticmethod
    def backward(ctx, grad):

        return grad, None, None,None


class Conv2d_Quantification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        groups: int = 1,
        bias: bool = True,
        timesteps: int =100):
        super().__init__()
        self.timesteps = timesteps
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
    
    def forward(self, x):
        x_q = Quantification.apply(x,self.timesteps)
        y = self.conv.forward(x_q)
        return y


class BatchNorm2d_Quantification(nn.Module):
    def __init__(self, filters, momentum=0.03, eps=1E-4, timesteps: int =100):
        super().__init__()
        self.timesteps = timesteps
        self.bn = nn.BatchNorm2d(filters, momentum, eps)

    def forward(self, x):
        x_q = Quantification.apply(x,self.timesteps)
        y = self.bn.forward(x_q)
        return y


class ConvTranspose2d_Quantification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups: int = 1,
        bias: bool = True,
        timesteps: int =100):
        super().__init__()
        self.timesteps = timesteps
        self.conv = nn.ConvTranspose2d(
            in_channels = in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias)
    
    def forward(self, x):
        x_q = Quantification.apply(x,self.timesteps)
        y = self.conv.forward(x_q)
        return y
