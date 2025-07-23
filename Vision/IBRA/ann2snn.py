import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

import torch
import torch.nn as nn
import torchvision
import time
import utils
import os
import model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from kdutils import seed_all
from tqdm import tqdm
import torchvision.transforms as transforms
import data

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
    
class Basicblocksnn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Basicblocksnn, self).__init__()
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
  
        )
        self.sn1 = mem_update()
        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.sn2 = mem_update()
        if stride != 1 or in_planes != planes:
            self.shortcut = layer.SeqToANNContainer(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3,padding=
                          1,  stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.shortcut(x) + out
        out = self.sn2(out)
        return out

class spiking_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(spiking_ResNet, self).__init__()

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

        self.annconv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.annbn1 = nn.BatchNorm2d(self.inplanes)
        self.sn1 = mem_update()
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


        #self.snsn = neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.annmaxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.flatten = layer.SeqToANNContainer(nn.Flatten())
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            zero_init_blocks(self)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes 
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out = self.annconv1(x)
        out = self.annbn1(out)
        out.unsqueeze_(0)
        out = self.sn1(out)
        out = self.annmaxpool(out)

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
def _resnet(block, layers, **kwargs):
    model = spiking_ResNet(block, layers, **kwargs)
    return model
def zero_init_blocks(net: nn.Module):
    for m in net.modules():
        if isinstance(m, Basicblocksnn):
            nn.init.constant_(m.conv2.module[1].weight, 0)

def resnet18(**kwargs):

    return _resnet(Basicblocksnn, [2, 2, 2, 2], **kwargs) 


def resnet34(**kwargs):

    return _resnet(Basicblocksnn, [3, 4, 6, 3], **kwargs) 


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

dist.init_process_group(backend='nccl', init_method='env://')

traindir = '/data/yebinghao//datasets/Imagenet/ILSVRC/Data/CLS-LOC/train'
testdir = '/data/yebinghao//datasets/Imagenet/ILSVRC/Data/CLS-LOC/val_gene'

batch_size = 512 # 每张卡的batchsize 
seed_all(666)
cache_dataset = True
distributed = True
num_epochs = 320


dataset_train, dataset_test, train_sampler, test_sampler = data.load_data(traindir, testdir,
                                                                cache_dataset, distributed)
#dataset_test, test_sampler = load_data(traindir, testdir,
#                                                 cache_dataset, distributed)
print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')



# 根据进程的排名计算局部 GPU ID
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f'cuda:{local_rank}')


train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size,
    sampler=train_sampler,  pin_memory=True, num_workers = 8)

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, pin_memory=True, num_workers = 8)


# model = model.spiking_resnet34(T=1).to(device)
model = resnet34().to(device)
# model = resnet.resnet18().to(device)


model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(model)
checkpoint = torch.load('/data/yebinghao/spiking_resnet/imagenet/best_validation_resnet34_seed666_imagenet_batch2048_1106.pth')
model.load_state_dict(checkpoint['model_state'])

loss_fun = nn.CrossEntropyLoss()
# 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cos_lr_T)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
a = []
ac_list = []

restart = False 
if restart:
    state = torch.load('model_checkpoint_ybh2.pth')
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    start_epoch = state['epoch']   
    scheduler = state['scheduler']
else:
    start_epoch = 0 
best_accuracy = 0.0  # 用于跟踪最佳验证准确率
best_model_state = None  # 用于保存最佳模型状态

scaler = GradScaler()  # 混合精度


model.eval()
total2 = torch.tensor(0, device=device)
correct2 = torch.tensor(0, device=device)
test_loss = torch.tensor(0.0, device=device)


if torch.distributed.get_rank() == 0:
    test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {1}/{num_epochs} [Test]")
else:
    test_loader_tqdm = test_loader
t2 = time.time()
with torch.no_grad():           
# 测试循环
    test_loss = 0
    correct = 0
    for data, target in test_loader_tqdm:
        data, target = data.to(device), target.to(device)

        output = model(data)
        test_loss += loss_fun(output, target).detach()  # 累加损失
        pred = output.data.max(1, keepdim=True)[1]  # 获取最大概率的索引
        correct2 += pred.eq(target.data.view_as(pred)).sum()
        total2 += target.size(0)
        if torch.distributed.get_rank() == 0:
            test_loader_tqdm.set_postfix({'Loss': (test_loss / total2).item(), 'Acc': (100. * correct2 / total2).item()})
    torch.distributed.all_reduce(test_loss)
    torch.distributed.all_reduce(correct2)
    torch.distributed.all_reduce(total2)

    test_loss /= total2
    test_accuracy = 100. * correct2 / total2
    t3 = time.time()

if torch.distributed.get_rank() == 0:
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Time: {:.4f}'.format(test_loss.item(), test_accuracy.item(), t3-t2))
