import torch
import numpy as np
import torch.nn as nn
from CNN_0402 import CNN_MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import logging
import os
import torch.optim as optim
from torch.optim import Adam, SGD
from numpy.random import RandomState
import torch.nn.functional as F
from sklearn.decomposition import MiniBatchDictionaryLearning
import torchvision.transforms.functional as transform_F
from scipy.io import savemat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SimCLRLoss2(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(torch.abs(similarity_matrix) / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, batch_size, CNN_out_dim, lambd):
        super().__init__()
        self.batch_size = batch_size
        self.CNN_out_dim =CNN_out_dim
        self.lambd = lambd

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # empirical cross-correlation matrix
        c = z_i.T @ z_j

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


# 数据增强函数
def rotate_img(image, angle):
    rotated_image = torch.zeros(image.shape)
    for i in range(image.shape[0]):
        rotated_image[i, :] = transform_F.rotate(image[i, :], angle)
    return rotated_image


def norm_data(data):
    data = data / np.max(np.abs(data))
    return data


batch_size = 64

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])  # 转为tensor，并归一化至[0-1]
data_path = "D:/program/NMNIST_data/"
train_dataset = datasets.MNIST(root=data_path,
                               train=True,
                               download=True,
                               transform=transform
                               )
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size,
                          drop_last=True
                          )
test_dataset = datasets.MNIST(root=data_path,
                              train=False,
                              download=True,
                              transform=transform
                              )
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size,
                         drop_last=True
                         )


class SC_dictionary_learning(torch.autograd.Function):

    @staticmethod
    def forward(ctx, outputs1, out_dim, dict_fit):
        dict_learner1 = MiniBatchDictionaryLearning(n_components=out_dim, alpha=0.1, batch_size=8,
                                                    dict_init=dict_fit)
        dc_input1 = outputs1.clone().detach().numpy()
        dict_fit = dict_learner1.fit(dc_input1).components_
        sparse_code1 = torch.from_numpy(dict_learner1.fit_transform(dc_input1))
        D_tensor = torch.from_numpy(dict_fit)
        reconstructed_signal1 = sparse_code1 @ D_tensor
        return reconstructed_signal1, dict_fit

    @staticmethod
    def backward(ctx, grad_outputs, grad_D):
        return grad_outputs, None, None, None


''' 更改 '''
model = CNN_MNIST()

# 在这里加入两行代码，将数据送入GPU中计算！！！
device = torch.device("cpu")
model.to(device)  # 将模型的所有内容放入cuda中

# 设置损失函数和优化器
# criterion = nn.MSELoss() # MSE
# criterion = nn.CrossEntropyLoss() # CEL
criterion = SimCLRLoss2(batch_size)  # CL2
# criterion = BarlowTwins(batch_size, CNN_out_dim=32, lambd=0.005)  # CL2
# 神经网络已经逐渐变大，需要设置冲量momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

logging.basicConfig(filename='./log/CNN32_D15_SimCLR.log', level=logging.INFO)

# 训练
# 将一次迭代封装入函数中
running_loss = 0.0


# 训练
# 将一次迭代封装入函数中
def train(epoch):
    global D
    running_loss = 0.0
    out_dim = 100
    CNN_out_dim = 32
    if epoch == 0:
        D = None

    for batch_idx, data in enumerate(train_loader, 0):  # 在这里data返回输入:inputs、输出target
        inputs1, labels = data
        # 在这里加入一行代码，将数据送入GPU中计算！！！
        inputs1, labels = inputs1.to(device), labels.to(device)
        inputs2 = rotate_img(inputs1, 15)
        # inputs2 = rotate_img(inputs1, 15)
        optimizer.zero_grad()

        # 前向 + 反向 + 更新
        outputs1 = model(inputs1)
        outputs2 = model(inputs2)
        outputs1 = torch.reshape(outputs1, (batch_size, CNN_out_dim))
        outputs2 = torch.reshape(outputs2, (batch_size, CNN_out_dim))
        rng = RandomState(0)
        reconstructed_signal1, D = SC_dictionary_learning.apply(outputs1, out_dim, D)
        reconstructed_signal2, D = SC_dictionary_learning.apply(outputs2, out_dim, D)

        loss = criterion.forward(reconstructed_signal2, reconstructed_signal1)
        print(loss.data)
        loss.backward()
        optimizer.step()

        if batch_idx == 936:
            save_dict1 = {'name': 'matrix', 'data': D}
            D_name = 'D:/program/model/epoch' + str(epoch) + '_dictionary'
            savemat(D_name, save_dict1)

        running_loss += loss.item()
        logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / (batch_idx + 1)))
        if batch_idx % 100 == 99 or batch_idx == 936:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / (batch_idx + 1)))
            path = 'D:/program/model'
            save_name = path + '/epoch' + str(epoch) + '_' + str(batch_idx) + '_CNN32_D15_SimCLR'
            torch.save(model.state_dict(), save_name)
    return D


for epoch in range(0, 100):
    train(epoch)
    path = 'D:/program/model'
    save_name = path + '/epoch' + str(epoch) + '_CNN32_D15_SimCLR'
    torch.save(model.state_dict(), save_name)
