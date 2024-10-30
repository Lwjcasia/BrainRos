import torch
import numpy as np
from CNN_sparse import CNN_FC, CNN
from SOM_traceSTDP_LI2 import SOM_STDP
# from SOM_classical import SOM
# from SOM_STDP_LateralInhibition import SOM_STDP
from snntorch import spikegen
from snntorch import utils
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.models import resnet18
import logging
import os
from scipy.io import savemat
from numpy.random import RandomState

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import random
import copy
from PiCCL import twins
from PIL import Image
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def rate_encoder(img_data, time_window):
    spike_data = spikegen.rate(img_data, num_steps=time_window)
    return spike_data


def eta_decya(eta, Tau_eta, data_num):
    return eta * np.exp(-data_num / Tau_eta)


def max_dist_decay(maxdist, Tau_maxdist, data_num):
    return maxdist * np.exp(-data_num / Tau_maxdist)


def sig_dist_decay(sig_dist, Tau_sigdist, data_num):
    return sig_dist * np.exp(-data_num / Tau_sigdist)


def heatmap(tallycount, epoch, data_num, mode='plot'):
    for i, feature in enumerate(tallycount):
        plt.figure()
        sns.heatmap(feature)
        plt.title(f"Iteration = {data_num}, lable = {i}")
        if mode == 'plot':
            plt.show()
        elif mode == 'save':
            plt.savefig(f'./results/images/{dataset}/{epoch}_{data_num}_{i}.png', bbox_inches='tight', dpi=300)
            plt.close()
    return


def train(epoch):
    tally = np.zeros((10, 40, 40), dtype=int)
    loaderbar = tqdm(train_loader)
    for batch, (img, label) in enumerate(loaderbar):
        img = img.to(device)

        CNN_out = model.forward(img)
        CNN_outdata = torch.reshape(CNN_out, (batch_size, CNN_out.shape[1]))
        
        weight_updtknl = net.weight_update_kernel(A, B, max_distance, sig_dist)
        for i in range(batch_size):
            spike_img_data = rate_encoder(CNN_outdata[i, :], time_window)
            # print(spike_img_data)
            data_num = batch * batch_size + i + 1
            if data_num % 200 == 199:
                weight_updtknl = net.weight_update_kernel(A, B, max_dist_decay(max_distance, (epoch*len_train_set + 2000)*SDF, data_num),
                                                          sig_dist_decay(sig_dist, (epoch*len_train_set + 2000)*SDF, data_num))
            eta_t = eta_decya(eta, 2000, data_num)
            O_tn, W, winner_t, winner_index = net.forward(CNN_outdata[i, :], tau_LTP, tau_LTD, eta_t, weight_updtknl)
            # O_tn, W, winner_t, winner_index = net.forward(img[i][0].reshape(-1), tau_LTP, tau_LTD, eta_t, weight_updtknl)
            tally[label[i], winner_index[0], winner_index[1]] += 1
            #print(data_num, winner_t, winner_index, label[i])
            if data_num % 2000 == 1999:
                '''
                print('Epoch %d, %5d, label %d, Winner firing time %d, Winner neuron index [%d, %d] ' % (
                    epoch + 1, data_num, label[i], winner_t[winner_index[0], winner_index[1]], winner_index[0] + 1,
                    winner_index[1] + 1))
                logging.info('Epoch %d, %5d, label %d, Winner firing time %d, Winner neuron index [%d, %d] ' % (
                    epoch + 1, data_num, label[i], winner_t[winner_index[0], winner_index[1]], winner_index[0] + 1,
                    winner_index[1] + 1))'''
                tally2 = copy.deepcopy(tally)
                save_dict1 = {'name': 'matrix', 'data': W}
                W_path = './results/SOM/' + str(epoch + 1) + '_' + str(data_num) + '.mat'  # 结果存储路径
                savemat(W_path, save_dict1)
                save_dict2 = {'name': 'matrix', 'data': tally}
                T_path = './results/SOM/' + 'tally' + str(epoch + 1) + '_' + str(data_num) + '.mat'  # 结果存储路径
                savemat(T_path, save_dict2)
                heatmap(tally2, epoch, data_num, mode='save')
                tally = np.zeros((10, 40, 40), dtype=int)
                print('HAHAHA')


def testSOM(noisesig):
    realvec = np.random.rand(10, 10)
    net = SOM()
    for i in range(10):
        realvec[i] = realvec[i] / np.linalg.norm(realvec[i])

    def samplegen():
        label = int(random.random() // .1)
        sample = realvec[label] + np.random.randn(10) * noisesig
        return label, sample

    numsample = 10000
    tally = np.zeros((10, 40, 40))
    weight_updtknl = net.weight_update_kernel(A, B, max_distance, 5)
    for i in range(numsample):
        eta_t = eta_decya(eta, (epoch*len_train_set + 2000)*SDF, i)
        label, sample = samplegen()
        O_tn, W, winner_t, winner_index = net.forward(sample, tau_LTP, tau_LTD, eta_t, weight_updtknl)
        tally[label, winner_index[0], winner_index[1]] += 1
        # print(i, winner_index, label)
        if i % 500 == 499:
            save_dict1 = {'name': 'matrix', 'data': W}
            W_path = './results/SOM/' + str(i) + '.mat'  # 结果存储路径
            savemat(W_path, save_dict1)
            print('HAHAHA')
        if i % 500 == 499:
            for i in range(10):
                plt.figure()
                sns.heatmap(tally[i], cmap='Oranges', vmin=0)
                plt.show()
            tally = np.zeros(tally.shape)


def adjust_state_dict_keys(state_dict, prefix_to_remove='enc', delete_keyword='projector',delete_keyword2='fc'):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(delete_keyword + '.'):
            continue
        elif key.startswith(delete_keyword2 + '.'):
            continue
        else:
            if key.startswith(prefix_to_remove + '.'):
                new_key = key[len(prefix_to_remove + '.'):]
            else:
                new_key =  key
            new_state_dict[new_key] = value
    return new_state_dict

if __name__ == "__main__":

    dataset='CIFAR10' # CIFAR10 or STL10

    # Define a transform
    if dataset == 'CIFAR10':
        size=32
        data_path = "./Data/CIFAR" 
        transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])  # ,transforms.Normalize((0.1307,), (0.3081,))
        train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        path = './results/CNN/simclr_resnet18_epoch500.pt'
    elif dataset == 'STL10':
        size=96
        data_path = "./Data/STL10" 
        transform = transforms.Compose([transforms.Resize((96, 96)),transforms.ToTensor()])  # ,transforms.Normalize((0.1307,), (0.3081,))
        train_set = datasets.STL10(data_path, split='train', download=True, transform=transform)
        path = './results/CNN/simclr_resnet18_epoch500_2.pt'
    subset = 1
    train_set = utils.data_subset(train_set, subset)
    len_train_set = len(train_set)
    print(f"The size of train_set is {len(train_set)}")

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    train_epoch = 5
    time_window = 50
    SDF = 2 # slow down factor

    # 模型加载
    model = twins(resnet18, size=size)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    model = model.enc
    # net = SOM_STDP()
    net = SOM_STDP(in_number=512)
    tau_LTP = 1
    tau_LTD = 1
    A = 1
    B = 0
    eta = 0.1/SDF
    max_distance = 10
    sig_dist = 5


    rng = RandomState(0)
    for epoch in range(train_epoch):
        train(epoch)
    # testSOM(0.1)





