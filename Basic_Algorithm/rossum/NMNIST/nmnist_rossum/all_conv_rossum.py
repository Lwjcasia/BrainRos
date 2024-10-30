import torch
import torch.nn as nn
import numpy as np
# from spikingjelly.clock_driven import encoding
from spikingjelly.clock_driven import neuron, functional


#cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
conv1=[2,12,1,1,5]
conv2=[12,64,1,0,5]

linear1=[2304,10]

time_window=25
tau_s=2.5

scale=1

# define approximate firing function   添加了STDP
# tau_syn = 5e-3
# time_step = 1e-3
# alpha = float(np.exp(-time_step/tau_syn))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,thresh):
        ctx.save_for_backward(input,thresh)
        # input1=input.gt(thresh)
        input1=input.gt(thresh)
        return input1.float()

    @staticmethod
    def backward(ctx, grad_output):
        input,thresh = ctx.saved_tensors
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

        spike=act_fun(mem_thr,thresh_i)
        mem = mem_thr - spike*thresh_i

        return mem, spike,psp



class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.thresh = torch.Tensor([ 0.7, 0.7, 0.7, 0.7, 0.7])
        self.decay = torch.Tensor([ 0.8, 0.8, 0.8, 0.8, 0.8])

        in_planes, out_planes, stride, padding, kernel_size = conv1
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)  #3-96 [32 32]

        in_planes, out_planes, stride, padding, kernel_size = conv2
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)  #96-256  [32 32]


        self.maxpool1=nn.MaxPool2d(2)   #256  [16 16]
        self.f1 = nn.Flatten()
        in_planes, out_planes = linear1
        self.linear1 = nn.Linear(in_features=in_planes, out_features=out_planes)  #1024
        # self.dropout6 = nn.Dropout(0.2)



    def forward(self,input):
        mem_input=torch.zeros(input.shape,device=device)

        psp_1=torch.zeros(input.shape[0],input.shape[1],input.shape[2],input.shape[3], device=device)
        mem1=torch.zeros((input.shape[0],conv1[1],32,32),device=device)


        psp_2=torch.zeros((input.shape[0], conv1[1], 16, 16), device=device)
        mem2 = torch.zeros((input.shape[0], conv2[1], 12, 12), device=device)

        psp_3 = torch.zeros((input.shape[0], linear1[0]), device=device)

        mem3 = torch.zeros((input.shape[0], linear1[1]), device=device)
        spike3=torch.zeros((time_window, input.shape[0], linear1[1]), device=device)
###########avgpool
        for t in range(time_window):


            mem1,spike1,psp_1=mem_update(self.conv1,psp_1,input[...,t],mem1,self.thresh[0],self.decay[0])
            pool1 = self.maxpool1(spike1)
            mem2,spike2,psp_2=mem_update(self.conv2,psp_2,pool1,mem2,self.thresh[1],self.decay[1])
            pool2 = self.maxpool1(spike2)


            zhanping=self.f1(pool2)

            mem3,spike3[t],psp3=mem_update(self.linear1,psp_3,zhanping,mem3,self.thresh[4],self.decay[4])


        # sum=torch.sum(spike5,dim=0)  #对这一批中每个样本的在时间窗口内的放电个数求和 输出shape[batchsize，10]
        # val,neuron_index=torch.max(sum,dim=1)   #找到每个样本放电最频繁的神经元的位置  输出shape[batchsize,1]
        # batch_index=torch.arange(0,spike5.shape[-2])
        # max_spike=spike5[:,batch_index,neuron_index]  #返回的是每个样本放电最频繁的神经元的脉冲串情况  输出shape[time_window,batchsize]

        return spike3

