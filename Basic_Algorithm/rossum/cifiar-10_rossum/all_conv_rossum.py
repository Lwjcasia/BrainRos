import torch
import torch.nn as nn
import numpy as np
# from spikingjelly.clock_driven import encoding
from spikingjelly.clock_driven import neuron, functional


#cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
conv1=[3,96,1,1,3]
conv2=[96,256,1,1,3]
conv3=[256,384,1,1,3]
conv4=[384,384,1,1,3]
conv5=[384,256,1,1,3]
linear1=[256*8*8,1024]
linear2=[1024,1024]
linear3=[1024,10]


time_window=10

# decay=1
thresh=1
tau_s=3


scale=1

# define approximate firing function   添加了STDP
# tau_syn = 5e-3
# time_step = 1e-3
# alpha = float(np.exp(-time_step/tau_syn))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,thresh):
        ctx.save_for_backward(input)
        # input1=input.gt(thresh)
        input1=input.gt(thresh)
        return input1.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #print('grad_out:',grad_input)
        # temp=1-torch.pow(nn.Tanh()(input),2)  #tanh函数的导数
        # temp=a*torch.sigmoid(a*input)*(1-torch.sigmoid(a*input))  #sigmod(ax) a=1.2的导数
        grad = grad_input / (scale * torch.abs(input - thresh) + 1.) ** 2
        #temp=abs(input - thresh) < 0.5
        #print('temp:', temp)
        return grad.float(),None

act_fun = ActFun.apply


def topsp(pre_psp,spike):
    # psp=alpha*pre_psp+spike
    psp=((1-1/tau_s)*pre_psp+spike)/tau_s
    return psp


# membrane potential update
# def mem_update(ops, pre_spike, mem,delta_time):
def mem_update(ops,pre_psp, pre_spike, mem,thresh_i,decay_i,drop=None):
        if drop!=None:
           pre_spike=drop(pre_spike)
        psp=topsp(pre_psp,pre_spike)
        # psp=alpha*pre_psp +pre_spike

        nn_spike=ops(psp)

        if ops.__class__.__name__=='Conv2d':
            nn_spike=nn.BatchNorm2d(nn_spike.shape[1]).cuda()(nn_spike)

        mem_thr = decay_i * mem + nn_spike
        # else:
        #     mem_thr =decay*mem+nn_spike
        # spike = act_fun(mem_thr,delta_time)  # act_fun : approximation firing function
        spike=act_fun(mem_thr,thresh_i)
        mem = mem_thr - spike*thresh_i

        return mem, spike,psp


def mem_update_nospike(ops, pre_psp,pre_spike, mem, decay_i):
    # psp = alpha * pre_psp + pre_spike
    psp = topsp(pre_psp, pre_spike)
    nn_spike = ops(psp)

    mem_thr = decay_i * mem + nn_spike
    return mem_thr,psp

def lif_spike_update(input,mem,decay_i=0.2,thresh_i=1): #0.5 0.4

    mem_ =  decay_i*mem +input
    spike=act_fun(mem_,thresh_i)

    mem=mem_-spike*thresh_i

    return mem,spike


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.thresh = torch.Tensor([ 1, 1, 1, 1, 1])
        self.decay = torch.Tensor([ 0.8, 0.8, 0.8, 0.8, 0.8])

        in_planes, out_planes, stride, padding, kernel_size = conv1
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)  #3-96 [32 32]

        self.dropout1 = nn.Dropout2d(0.2)

        in_planes, out_planes, stride, padding, kernel_size = conv2
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)  #96-256  [32 32]


        self.maxpool1=nn.MaxPool2d(2)   #256  [16 16]
        self.dropout2 = nn.Dropout2d(0.2)


        in_planes, out_planes, stride, padding, kernel_size = conv3
        self.conv3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)    ##256-384  [16 16]
        self.maxpool2 = nn.MaxPool2d(2)  #384  [8 8]
        self.dropout3 = nn.Dropout2d(0.2)

        in_planes, out_planes, stride, padding, kernel_size = conv4
        self.conv4 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)   ##384-384  [8 8]
        self.dropout4 = nn.Dropout2d(0.2)

        in_planes, out_planes, stride, padding, kernel_size = conv5
        self.conv5 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)    ##384-256  [8 8]

        self.dropout5 = nn.Dropout2d(0.2)

        self.f1 = nn.Flatten()
        in_planes, out_planes = linear1
        self.linear1 = nn.Linear(in_features=in_planes, out_features=out_planes)  #1024
        self.dropout6 = nn.Dropout(0.2)

        in_planes, out_planes = linear2
        self.linear2 = nn.Linear(in_features=in_planes, out_features=out_planes)  #1024
        self.dropout7 = nn.Dropout(0.2)

        in_planes, out_planes = linear3
        self.linear3 = nn.Linear(in_features=in_planes, out_features=out_planes)  #10

    def forward(self,input):
        mem_input=torch.zeros(input.shape,device=device)
        # psp_input = torch.zeros(input.shape, device=device)

        psp_1=torch.zeros(input.shape, device=device)
        mem1=torch.zeros((input.shape[0],conv1[1],32,32),device=device)
        # spike1=torch.zeros((time_window,input.shape[0],conv1[1],32,32),device=device)

        psp_2=torch.zeros((input.shape[0], conv1[1], 32, 32), device=device)
        mem2 = torch.zeros((input.shape[0], conv2[1], 32, 32), device=device)
        # spike2 = torch.zeros((time_window, input.shape[0], conv2[1], 16, 16), device=device)

        psp_3 = torch.zeros((input.shape[0], conv2[1], 16, 16), device=device)
        mem3 = torch.zeros((input.shape[0], conv3[1], 16, 16), device=device)
        # spike3 = torch.zeros((time_window, input.shape[0], conv3[1], 16, 16), device=device)

        psp_4 = torch.zeros((input.shape[0], conv3[1], 8, 8), device=device)
        mem4 = torch.zeros((input.shape[0], conv4[1], 8, 8), device=device)

        psp_5 = torch.zeros((input.shape[0], conv4[1], 8, 8), device=device)
        mem5 = torch.zeros((input.shape[0], conv5[1], 8, 8), device=device)

        psp6=torch.zeros((input.shape[0],conv5[1]*8*8),device=device)
        mem6 = torch.zeros((input.shape[0], linear1[1]), device=device)
        # spike5 = torch.zeros((time_window, input.shape[0], linear1[1]), device=device)

        psp7 = torch.zeros((input.shape[0], linear1[1]), device=device)
        mem7 = torch.zeros((input.shape[0], linear2[1]), device=device)

        out_psp=torch.zeros((input.shape[0], linear2[1]), device=device)
        out_mem=torch.zeros((input.shape[0], linear3[1]), device=device)
        out_spike=torch.zeros((time_window, input.shape[0], linear3[1]), device=device)
###########avgpool
        for t in range(time_window):
            # mem_input,spike_input=lif_spike_update(input,mem_input)
            # psp_input=topsp(psp_input,spike_input)


            mem1,spike1,psp_1=mem_update(self.conv1,psp_1,input,mem1,self.thresh[0],self.decay[0])
            # sum1 = torch.sum(spike1, dim=0)

            mem2,spike2,psp_2=mem_update(self.conv2,psp_2,spike1,mem2,self.thresh[1],self.decay[1],self.dropout1)
            pool1 = self.maxpool1(spike2)

            mem3,spike3,psp_3=mem_update(self.conv3,psp_3,pool1,mem3,self.thresh[2],self.decay[2],self.dropout2)
            # sum3=torch.sum(spike3,dim=0)
            pool3 = self.maxpool2(spike3)


            mem4,spike4,psp_4=mem_update(self.conv4,psp_4,pool3,mem4,self.thresh[3],self.decay[3],self.dropout3)

            mem5, spike5, psp_5 = mem_update(self.conv5, psp_5, spike4, mem5, self.thresh[3], self.decay[3],self.dropout4)

            zhanping=self.f1(spike5)

            mem6,spike6,psp6=mem_update(self.linear1,psp6,zhanping,mem6,self.thresh[4],self.decay[4],self.dropout5)

            mem7, spike7, psp7 = mem_update(self.linear2, psp7, spike6, mem7, self.thresh[4], self.decay[4],self.dropout6)

            out_mem,out_spike[t],out_psp=mem_update(self.linear3,out_psp,spike7,out_mem,self.thresh[4],self.decay[4],self.dropout7)

        # sum=torch.sum(spike5,dim=0)  #对这一批中每个样本的在时间窗口内的放电个数求和 输出shape[batchsize，10]
        # val,neuron_index=torch.max(sum,dim=1)   #找到每个样本放电最频繁的神经元的位置  输出shape[batchsize,1]
        # batch_index=torch.arange(0,spike5.shape[-2])
        # max_spike=spike5[:,batch_index,neuron_index]  #返回的是每个样本放电最频繁的神经元的脉冲串情况  输出shape[time_window,batchsize]

        return out_spike
        # return spike5,out_spike
# #
# model = SCNN().cuda()
#
# inp = torch.rand(16, 1, 28, 28).cuda()
#
# outp = model(inp)
#
# print(outp.shape)
#
