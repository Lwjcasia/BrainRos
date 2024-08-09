import math
from spikingjelly.clock_driven import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor
from typing import Callable
from spikingjelly.clock_driven import functional
import pdb 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shift_module(x):
    # print("************shift_module: {}".format(x.size()))
    out = torch.zeros_like(x) 
    b, t, c, h, w = x.size() 
    out[:, :-1, :c//2] = x[:, 1:, :c//2]  # shift left
    # out[:, 1:, :c//2] = x[:, :-1, :c//2]  # shift right
    out[:, :, c//2:] = x[:, :, c//2:]  # not shift
    return out 

class LModule(nn.Module):
    def __init__(self, in_channels, mid_channels) -> None:
        super(LModule, self).__init__()

        self.conv_local1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        self.bn_local1 = nn.BatchNorm2d(mid_channels) 
        self.conv_local2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        self.bn_local2 = nn.BatchNorm2d(mid_channels) 
        self.conv_local3 = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1)
        self.bn_local3 = nn.BatchNorm2d(in_channels) 

        self.active = nn.Sigmoid() 

    def forward(self, x):
        features = shift_module(x) 

        features = functional.seq_to_ann_forward(features, [self.conv_local1, self.bn_local1]) 
        features = functional.seq_to_ann_forward(features, [self.conv_local2, self.bn_local2]) 
        features = functional.seq_to_ann_forward(features, [self.conv_local3, self.bn_local3]) 

        features = self.active(features) 
        out = features * x + x 
        # out = features * x 
        return out 
        # return features 

class GModule(nn.Module):
    def __init__(self, time, mode='permute') -> None:
        super(GModule, self).__init__()
        self.mode = mode 
        self.conv_long = nn.Conv2d(in_channels=time, out_channels=time, kernel_size=3, stride=1, padding=1)
        self.bn_long = nn.BatchNorm2d(time)
        self.active = nn.Sigmoid() 

    def forward(self, x):
        b, t, c, h, w = x.size() 
        features = x 
        if self.mode == 'cut': 
            # features, _ = torch.max(features, dim=2) # b, t, h, w
            features =  torch.mean(features, dim=2, keepdim=False)
            features = self.conv_long(features)
            features = self.bn_long(features)
            features = features.unsqueeze(2).repeat(1,1,c,1,1) # b, t, c, h, w 
        elif self.mode == 'per': 
            features = features.permute(0, 2, 1, 3, 4).contiguous().view(-1, t, h, w) # b, t, c, h, w -> b*c, t, h, w
            features = self.conv_long(features)
            features = self.bn_long(features)
            features = features.view(b, c, -1, h, w).permute(0, 2, 1, 3, 4).contiguous() # b*c, t, h, w -> b, c, t, h, w -> b, t, c, h, w 
        else:
            raise NotImplementedError(self.mode) 
        
        features = self.active(features) 
        out = features * x + x 
        # out = features * x 
        return out 
    
class LGModule(nn.Module):
    def __init__(self, in_channels, mid_channels, time, mode='per') -> None:
        super(LGModule, self).__init__()
        self.LM = LModule(in_channels, mid_channels)
        self.GM = GModule(time, mode)

    def forward(self, x):
        out = x
        out = self.LM(out) 
        out = self.GM(out)
        # x = x_l * x_g 
        return out 

class TAModule(nn.Module):
    def __init__(self, in_channels, mid_channels, time, mode='permute') -> None:
        super(TAModule, self).__init__()
        self.mode = mode 
        self.in_channels = in_channels 
        self.mid_channels = mid_channels 
        # self.conv_short = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        # self.bn_short = nn.BatchNorm2d(mid_channels)
        # if in_channels != mid_channels:
        #     self.conv_short_1 = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1)
        #     self.bn_short_1 = nn.BatchNorm2d(in_channels)

        self.conv_long = nn.Conv2d(in_channels=time, out_channels=time, kernel_size=3, stride=1, padding=1)
        self.bn_long = nn.BatchNorm2d(time)
        
        self.active = nn.Sigmoid() 

    def forward(self, x):
        # pdb.set_trace() 
        # features = cat_channels(x) 
        # features = shift_module(x) 
        # features = functional.seq_to_ann_forward(features, [self.conv_short, self.bn_short]) 
        # if self.in_channels != self.mid_channels: 
        #     features = functional.seq_to_ann_forward(features, [self.conv_short_1, self.bn_short_1])
        features = x 
        b, t, c, h, w = features.size() 

        if self.mode == 'cut': 
            features, _ = torch.max(features, dim=2) # b, t, h, w
            # features = features.transpose(0, 1).contiguous() # b, t, h, w
            features = self.conv_long(features)
            features = self.bn_long(features)
            features = features.unsqueeze(2).repeat(1,1,c,1,1) # b, t, c, h, w 
            # features = features.transpose(0, 1).contiguous().unsqueeze(2).repeat(1,1,c,1,1) # t, b, c, h, w 
        elif self.mode == 'permute': 
            features = features.permute(0, 2, 1, 3, 4).contiguous().view(-1, t, h, w) # b, t, c, h, w -> b*c, t, h, w
            features = self.conv_long(features)
            features = self.bn_long(features)
            features = features.view(b, c, -1, h, w).permute(0, 2, 1, 3, 4).contiguous() # b*c, t, h, w -> b, c, t, h, w -> b, t, c, h, w 
        else:
            raise NotImplementedError(self.mode) 
        
        features = self.active(features) 
        out = features * x + x 
        return out 
    
class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        # x = self.act(x)
        return x

class TEBN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        self.p = nn.Parameter(torch.ones(10, 1, 1, 1, 1, device=device))
    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y

class TEBNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        # x = self.act(x)
        return y

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.pre_spike_mem = []

    def forward(self, x):
        mem_v = []
        _mem = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            _mem.append(mem.detach().cpu().clone())
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)
        self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(mem_v, dim=1)

class MaskedSlidingPSN(nn.Module):

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight


    def __init__(self, order: int = 2, surrogate_function = surrogate.ATan(), exp_init: bool=True):
        super().__init__()

        self.order = order
        if exp_init:
            weight = torch.ones([order])
            for i in range(order - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.

            self.weight = nn.Parameter(weight)
        else:
            self.weight = torch.ones([1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight[0])
        self.threshold = nn.Parameter(torch.as_tensor(-1.))
        self.surrogate_function = surrogate_function


    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [N, T, *]
        weight = self.gen_gemm_weight(x_seq.shape[1])
        h_seq = F.linear(x_seq.transpose(1, -1), weight, self.threshold)
        h_seq = h_seq.transpose(1, -1)
        
        return self.surrogate_function(h_seq)

class VGGPSN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGPSN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)

        # tam parameters
        reduction = 32 # lm reduction 
        time = 10 # time 
        gm_mode = 'cut' # gm mode 

        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            LGModule(64, 64//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            Layer(64, 128, 3, 1, 1),
            LGModule(128, 128//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            pool,
            # LGModule(128, 128//reduction, time, gm_mode),
            Layer(128, 256, 3, 1, 1),
            LGModule(256, 256//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            Layer(256, 256, 3, 1, 1),
            LGModule(256, 256//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            pool,
            # LGModule(256, 256//reduction, time, gm_mode),
            Layer(256, 512, 3, 1, 1),
            LGModule(512, 512//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            Layer(512, 512, 3, 1, 1),
            LGModule(512, 512//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            pool,
            # LGModule(512, 512//reduction, time, gm_mode),
            Layer(512, 512, 3, 1, 1),
            LGModule(512, 512//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            Layer(512, 512, 3, 1, 1),
            LGModule(512, 512//reduction, time, gm_mode),
            MaskedSlidingPSN(),
            pool,
            # LGModule(512, 512//reduction, time, gm_mode),
        )
        W = int(48 / 2 / 2 / 2 / 2) 
        # self.T = 10
        self.classifier = nn.Sequential(SeqToANNContainer(nn.Dropout2d(0.25), nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)  # [N, T, C, H, W]
        # input = add_dimention(input, self.T)
        # pdb.set_trace()

        # original code
        x = self.features(input)
        x = torch.flatten(x, 2)
        out = self.classifier(x)
        return out

        # firing number version
        # pdb.set_trace()
        # firing_num = 0
        # out = input
        # for conv_layer in self.features:
        #     out = conv_layer(out)
        #     if isinstance(conv_layer, MaskedSlidingPSN):
        #         firing_num += out.sum().item()
        #         print("outsize:{}, firing_num:{}".format(out.size(), (out>0).sum().item()))
        #         # pdb.set_trace()
        # out = torch.flatten(out, 2) 
        # out = self.classifier(out) 
        # return out #, firing_num 

class VGGSNN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGSNN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(64, 128, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(256, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(512, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x