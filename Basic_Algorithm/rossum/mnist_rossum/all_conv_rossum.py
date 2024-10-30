import torch
import torch.nn as nn
import numpy as np
# from spikingjelly.clock_driven import encoding
from spikingjelly.clock_driven import neuron, functional


#cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
conv1=[1,12,1,1,3]
conv2=[12,64,1,1,3]
conv3=[64,128,1,1,3]
conv4=[128,256,1,1,3]
linear=[256*9,10]

time_window=10
a=1.5
decay=1
thresh=1
tau_s=2


scale=1

# define approximate firing function   添加了STDP
# tau_syn = 5e-3
# time_step = 1e-3
# alpha = float(np.exp(-time_step/tau_syn))

alpha=0.5

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
def mem_update(ops, pre_psp, pre_spike, mem,thresh_i,decay_i):
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

def lif_spike_update(input,mem,decay_i=0.8,thresh_i=1): #0.5 0.4

    mem_ =  decay_i*mem +input
    spike=act_fun(mem_,thresh_i)

    mem=mem_-spike*thresh_i

    return mem,spike
#
# def find_maxindx(sumi,spike):
#
#     maxpool,index=nn.MaxPool2d(2,return_indices=True)(sumi)
#
#     weight = sumi.shape[3]
#
#     x = (index / weight).long()
#     y = (index % weight).long()
#     a = torch.zeros((index.shape[0], index.shape[1], maxpool.shape[-2], maxpool.shape[-1]))
#     b = torch.zeros((index.shape[0], index.shape[1], maxpool.shape[-2], maxpool.shape[-1]))
#     c = torch.zeros((index.shape[0], index.shape[1], maxpool.shape[-2], maxpool.shape[-1]))
#     d = torch.zeros((index.shape[0], index.shape[1], maxpool.shape[-2], maxpool.shape[-1]))
#
#
#     for batch in range(index.shape[0]):
#         for channel in range(index.shape[1]):
#             # out[batch,channel,:,:]=mem[batch,channel,x[batch,channel,:],y[batch,channel,:]]
#
#             a[batch, channel] = batch
#
#             b[batch, channel] = channel
#
#             c[batch, channel] = x[batch, channel, :]
#
#             d[batch, channel] = y[batch, channel, :]
#     a=a.long()
#     b=b.long()
#     c=c.long()
#     d=c.long()
#
#     return spike[a,b,c,d]



class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.thresh = torch.Tensor([ 1, 1, 1, 1, 1])
        self.decay = torch.Tensor([ 0.8, 0.8, 0.8, 0.8, 0.8])

        in_planes, out_planes, stride, padding, kernel_size = conv1
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        self.maxpool1=nn.MaxPool2d(2)

        in_planes, out_planes, stride, padding, kernel_size = conv2
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)


        in_planes, out_planes, stride, padding, kernel_size = conv3
        self.conv3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)

        self.maxpool3 = nn.MaxPool2d(2)

        in_planes, out_planes, stride, padding, kernel_size = conv4
        self.conv4 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        


        self.f1 = nn.Flatten()
        in_planes, out_planes = linear
        self.linear = nn.Linear(in_features=in_planes, out_features=out_planes)


    def forward(self,input):
        mem_input=torch.zeros(input.shape,device=device)
        # psp_input = torch.zeros(input.shape, device=device)

        psp_1=torch.zeros(input.shape, device=device)
        mem1=torch.zeros((input.shape[0],conv1[1],28,28),device=device)
        spike1=torch.zeros((time_window,input.shape[0],conv1[1],28,28),device=device)

        psp_2=torch.zeros((input.shape[0], conv1[1], 14, 14), device=device)
        mem2 = torch.zeros((input.shape[0], conv2[1], 14, 14), device=device)
        spike2 = torch.zeros((time_window, input.shape[0], conv2[1], 14, 14), device=device)

        psp_3 = torch.zeros((input.shape[0], conv2[1], 7, 7), device=device)
        mem3 = torch.zeros((input.shape[0], conv3[1], 7, 7), device=device)
        spike3 = torch.zeros((time_window, input.shape[0], conv3[1], 7, 7), device=device)

        psp_4 = torch.zeros((input.shape[0], conv3[1], 3, 3), device=device)
        mem4 = torch.zeros((input.shape[0], conv4[1], 3, 3), device=device)


        psp_5=torch.zeros((input.shape[0],conv4[1]*3*3),device=device)
        mem5 = torch.zeros((input.shape[0], linear[1]), device=device)
        spike5 = torch.zeros((time_window, input.shape[0], linear[1]), device=device)

        out_mem=torch.zeros((input.shape[0], linear[1]), device=device)
        out_spike=torch.zeros((time_window, input.shape[0], linear[1]), device=device)
###########avgpool
        for t in range(time_window):
            mem_input,spike_input=lif_spike_update(input,mem_input)
            # psp_input=topsp(psp_input,spike_input)


            mem1,spike1[t],psp_1=mem_update(self.conv1,psp_1,spike_input,mem1,self.thresh[0],self.decay[0])
            sum1=torch.sum(spike1,dim=0)

            pool1=self.maxpool1(sum1)


            mem2,spike2[t],psp_2=mem_update(self.conv2,psp_2,pool1/(t+1),mem2,self.thresh[1],self.decay[1])
            sum2 = torch.sum(spike2, dim=0)
            pool2 = self.maxpool2(sum2)

            mem3,spike3[t],psp_3=mem_update(self.conv3,psp_3,pool2/(t+1),mem3,self.thresh[2],self.decay[2])
            sum3=torch.sum(spike3,dim=0)
            pool3=self.maxpool3(sum3)


            mem4,spike4,psp_4=mem_update(self.conv4,psp_4,pool3/(t+1),mem4,self.thresh[3],self.decay[3])

            zhanping=self.f1(spike4)

            mem5,spike5[t],psp5=mem_update(self.linear,psp_5,zhanping,mem5,self.thresh[4],self.decay[4])

            # out_mem,out_spike[t]=lif_spike_update(label,out_mem)

        # sum=torch.sum(spike5,dim=0)  #对这一批中每个样本的在时间窗口内的放电个数求和 输出shape[batchsize，10]
        # val,neuron_index=torch.max(sum,dim=1)   #找到每个样本放电最频繁的神经元的位置  输出shape[batchsize,1]
        # batch_index=torch.arange(0,spike5.shape[-2])
        # max_spike=spike5[:,batch_index,neuron_index]  #返回的是每个样本放电最频繁的神经元的脉冲串情况  输出shape[time_window,batchsize]

        return spike5
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
