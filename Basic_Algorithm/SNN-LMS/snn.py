import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score
import time
import torch.nn.functional as F

print(torch.__version__)  # torch的版本

# 参数
nb_inputs = 28 * 28
nb_hidden = 100
nb_outputs = 10

time_step = 1e-3  # 时间步长为1ms
nb_steps = 100  # 共100个时间步长

batch_size = 200

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():  # GPU 能否被 PyTorch 调用
    device = torch.device("cuda")  # torch.device()是一个对象，表示分配torch.tensor的设备；设备类型有CPU或cuda
else:
    device = torch.device("cpu")  # 模型加载到指定设备
print(1, device)
# Here we load the Dataset
root = os.path.expanduser("E:/datasets/FashionMNIST")  # 将参数中的~替换为当前用户的home目录并返回；定义了数据的下载路径
train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None,
                                                  download=True)
test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None,
                                                 download=True)

# Standardize data
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.data, dtype=np.float64)
x_train = x_train.reshape(x_train.shape[0], -1) / 255  # 训练数据(60000，28，28) -> (60000,784)
# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.data, dtype=np.float64)
x_test = x_test.reshape(x_test.shape[0], -1) / 255  # 测试数据(10000,28,28) -> (10000,784)

# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.targets, dtype=int)  # 训练标签 (60000,)
y_test = np.array(test_dataset.targets, dtype=int)  # 测试标签 (10000,)


def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value
    tmax -- The maximum time returned
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x < thr  # 未达到阈值的为true
    x = np.clip(x, thr + epsilon, 1e9)  # numpy.clip(a, min, max, out=None):元素限制在min,max之间，大于max的就使其等于max，小于min的就其等于min
    T = tau * np.log(x / (x - thr))  # 未达到阈值的值最高,即一直未激发脉冲；达到阈值的值较低，表示较早的激发了脉冲
    T[idx] = tmax  # 未达到阈值的置为指定的值tmax
    return T


def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y, dtype=int)
    number_of_batches = len(X) // batch_size  # 有多少个batch
    sample_index = np.arange(len(X))
    # compute discrete firing times
    tau_eff = 20e-3 / time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=int)
    unit_numbers = np.arange(nb_units)
    if shuffle:
        np.random.shuffle(sample_index)  # 将序列的所有元素随机排序

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            # print(222, firing_times[idx])
            c = firing_times[idx] < nb_steps
            # print(333, c.shape, unit_numbers[c].shape)
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1


# Setup of the spiking network model

tau_mem = 10e-3  # 膜电位常数
tau_syn = 5e-3

alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))

weight_scale = 7 * (1.0 - beta)  # ？？？this should give us some spikes to begin with
#
# w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)  # 创建一个使用未初始化值填满的tensor
# torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(nb_inputs))  # 用给定均值和标准差的正态分布N(mean, std)中生成的值来填充输入的张量
#
# w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
# torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

print("init done")


# Training the network

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn = SurrGradSpike.apply


class LIF(nn.Module):
    def __init__(self, inp, oup, tau_m=tau_mem, tau_i=tau_syn, last=False, device=device):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.tm = tau_m
        self.ti = tau_i
        self.beta = float(np.exp(-time_step / tau_mem))
        self.weight_scale = 7 * (1.0 - self.beta)
        self.w = torch.empty((inp, oup), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.w, mean=0.0, std=weight_scale / np.sqrt(inp))
        self.w = nn.Parameter(self.w)
        self.alpha = float(np.exp(-time_step / tau_syn))
        self.last = last
        self.epsilon = torch.nn.Parameter(torch.FloatTensor([[[0.006, 0.121, 0.512, 0.121, 0.006]]]), requires_grad=True,).to(device)
    def forward(self, inputs):
        n, h, c = inputs.shape
        # print(111, inputs.shape)
        inputs_smooth = inputs.permute(0, 2, 1).reshape(n*c, 1, h)
        # print(122, inputs_smooth.shape)
        inputs_smooth = F.conv1d(inputs_smooth, self.epsilon, padding=2).reshape(n, c, h).permute(0, 2, 1)
        # print(222, inputs_smooth.shape)
        h1 = torch.einsum("abc,cd->abd", (inputs_smooth, self.w))
        syn = torch.zeros((n, self.oup), device=device, dtype=dtype)
        mem = torch.zeros((n, self.oup), device=device, dtype=dtype)
        if self.last:
            mem_rec = [mem]
        else:
            mem_rec = []
        spk_rec = []
        for t in range(nb_steps):
            mthr = mem - 1.0
            out = spike_fn(mthr)
            # rst = out.detach()  # We do not want to backprop through the reset
            rst = out

            new_syn = self.alpha * syn + h1[:, t]
            if self.last:
                new_mem = (self.beta * mem + syn)
            else:
                new_mem = (self.beta * mem + syn) * (1.0 - rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return mem_rec, spk_rec


class SNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lif1 = LIF(nb_inputs, nb_hidden)
        # self.lif2 = LIF(nb_hidden, nb_hidden)
        # self.lif3 = LIF(nb_hidden, nb_hidden)
        self.last = LIF(nb_hidden, nb_outputs, last=True)

    def forward(self, inputs):
        m1, s1 = self.lif1(inputs)
        # m2, s2 = self.lif2(s1)
        # m3, s3 = self.lif2(s2)
        m_last, s_lat = self.last(s1)
        return m_last, [m1, s1]


def train(x_data, y_data, a_model, lr=1e-3, nb_epochs=10):
    # params = [w1, w2]

    # a_model.train()
    optimizer = torch.optim.Adam(a_model.parameters(), lr=lr, betas=(0.9, 0.999))  # 构造一个优化器对象Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    loss_hist = []
    acc_hist = []
    for e in range(nb_epochs):
        local_loss = []
        acc_lst = []
        time0 = time.time()
        for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs):  # 生成器generator
            output, recs = a_model(x_local.to_dense())
            m, _ = torch.max(output, 1)
            _, spks = recs
            log_p_y = log_softmax_fn(m)
            reg_loss = 1e-5 * torch.sum(spks)  # L1 loss on total number of spikes
            reg_loss += 1e-5 * torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)
            loss_val = loss_fn(log_p_y, y_local.long()) + reg_loss
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
            acc_local = accuracy_score(y_local.detach().cpu().numpy(),
                                       log_p_y.argmax(1).detach().cpu().numpy())
            acc_lst.append(acc_local)
        mean_loss = np.mean(local_loss)
        mean_acc = np.mean(acc_lst)
        scheduler.step()

        print("Epoch %i: loss=%.5f, acc=%.4f, time=%.2f" % (e + 1, mean_loss, mean_acc, time.time() - time0))
        loss_hist.append(mean_loss)
        acc_hist.append(mean_acc)
    return loss_hist, acc_hist


def test(x_test, y_test, a_model, batch_size, nb_steps, nb_inputs):
    a_model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()

    with torch.no_grad():  # 在评估过程中不计算梯度
        for x_local, y_local in sparse_data_generator(x_test, y_test, batch_size, nb_steps, nb_inputs):
            output, recs = a_model(x_local.to_dense())
            m, _ = torch.max(output, 1)
            log_p_y = log_softmax_fn(m)
            test_loss += loss_fn(log_p_y, y_local.long()).item()
            pred = log_p_y.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_local.long().view_as(pred)).sum().item()

    test_loss /= len(x_test)
    accuracy = correct / len(x_test)

    print(f'Test set: Accuracy: {accuracy:.4f}%')


# def compute_classification_accuracy(x_data, y_data):
#     """ Computes classification accuracy on supplied data in batches. """
#     accs = []
#     for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs,
#                                                   shuffle=False):  # 生成器
#         output, _ = run_snn(x_local.to_dense())
#         m, _ = torch.max(output, 1)  # max over time
#         _, am = torch.max(m, 1)  # argmax over output units
#         tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
#         accs.append(tmp)
#     return np.mean(accs)


model = SNN_Model().to('cuda')
# loss_hist = train(x_train, y_train, lr=1e-3, nb_epochs=50)
loss_hist, acc_hist = train(x_train, y_train, model, lr=1e-3, nb_epochs=50)
test(x_test, y_test, model, batch_size, nb_steps, nb_inputs)

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_hist) + 1), loss_hist, marker='o', label='Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(acc_hist) + 1), acc_hist, marker='o', label='Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.show()
