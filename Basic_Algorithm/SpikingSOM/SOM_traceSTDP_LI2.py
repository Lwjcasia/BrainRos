# -*- coding: utf-8 -*-

# 原作者：陈霏 SOM_STDP_LateralInhibition.py

import copy
import numpy as np
import math
import torch
import torch.nn as nn
from numpy import array
#from spikingjelly.clock_driven import encoding
from snntorch import spikegen
from torch import tensor, reshape
from scipy import signal

PSP_kernal = [0, 0, 0, 0, 0, 0.5, 0.2789, 0.0458, 0]


def rate_encoder(img_data, time_window):
    img_data = img_data / max(img_data.reshape(-1))
    spike_data = spikegen.rate(img_data, num_steps=time_window)
    return spike_data


def poisson_encoder(img_data, time_window):
    layer1_input = torch.full((time_window, img_data.shape[0], img_data.shape[1]), 0, dtype=torch.bool)
    encoder = encoding.PoissonEncoder()
    for t in range(time_window):
        layer1_input[t] = encoder(img_data)

    m = nn.Flatten(1, 2)
    layer1_spikes: array = m(layer1_input)  # shape:tine_window,28*28
    return layer1_spikes


def psp(spikes):
    shape = spikes.shape
    syns = torch.zeros(shape)
    out_spikes = np.zeros(syns.shape)
    for k in range(shape[1]):
        in_spikes = spikes[..., k]
        out_spikes[..., k] = np.convolve(in_spikes.detach().numpy(), PSP_kernal, 'same')
    syns = torch.from_numpy(out_spikes)

    return syns


def distance(index1, index2):
    a = np.abs(index1[0] - index2[0])
    b = np.abs(index1[1] - index2[1])
    dis = math.sqrt(a * a + b * b)
    return dis


def gauss_function(x, y, sigma=1):
    value = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return value


def create_gauss_kernel(size=3, sigma=1):
    kernel = np.zeros((size, size))
    centre = size // 2
    for i in range(size):
        for j in range(size):
            x = i - centre
            y = j - centre
            kernel[i, j] = gauss_function(x, y, sigma)
    return kernel


def create_DOG_kernel(size=3, sigma1=1, sigma2=1.5):  # sigma越大曲线就越宽
    kernel1 = create_gauss_kernel(size, sigma1)
    kernel2 = create_gauss_kernel(size, sigma2)
    kernel = kernel1 - kernel2
    return kernel


def norm_W(W):
    for i in range(W.shape[1]):
        for j in range(W.shape[2]):
            W[:, i, j] = W[:, i, j] / np.linalg.norm(W[:, i, j])
    return W


class SOM_STDP(object):
    def __init__(self, in_number=128):
        self.iteration = 100
        self.batch_size = 64
        self.in_number = in_number
        self.output = np.zeros((40, 40))  # 竞争层节点的数量最少为5*（batch_size*iteration）^0.5，括号内为训练样本的数量
        self.time_window = 50

        self.tau_layer2 = 0.9
        self.V_peak = 15
        self.V_th = 100  # 放电阈值
        self.V_rest = 0  # 静息电位
        self.W = np.random.rand(self.in_number, self.output.shape[0], self.output.shape[1])
        self.W = norm_W(self.W)
        # self.RFP=1 # Refractory Period不应期

        self.input_norm_const = 50

    def weight_update_kernel(self, inweight, outweight, max_distance, sigma):
        axe1size = self.output.shape[0]
        axe2size = self.output.shape[1]
        quarter_wuk = np.zeros((axe1size, axe2size))
        for i in range(axe1size):
            for j in range(axe2size):
                if np.sqrt(i ** 2 + j ** 2) <= max_distance:
                    quarter_wuk[i, j] = inweight * np.exp(- (i ** 2 + j ** 2) / (2 * sigma ** 2))
                else:
                    quarter_wuk[i, j] = outweight * np.exp(- (i ** 2 + j ** 2) / (2 * sigma ** 2))
        half_wuk = np.concatenate((np.flip(quarter_wuk, 1)[:, :-1], quarter_wuk), 1)
        full_wuk = np.concatenate((np.flip(half_wuk, 0)[:-1, :], half_wuk), 0)
        return full_wuk
    
    def periodic_weight_update_kernel(self, inweight, outweight, max_distance, sigma):
        x = torch.zeros(self.output.shape[0])
        for i in range(self.output.shape[0]):
            x[i] = min(i,self.output.shape[0]-i)
        y = torch.zeros(self.output.shape[1])
        for i in range(self.output.shape[1]):
            y[i] = min(i,self.output.shape[1]-i)
        x = x.expand(*self.output.shape)
        y = y.expand(self.output.shape[1],self.output.shape[0]).T
        quarter_wuk = torch.exp(torch.add(x.pow(2),y.pow(2))/(-2 * max_distance**2))
        half_wuk = torch.concatenate((torch.flip(quarter_wuk,[1])[:,:-1], quarter_wuk), 1)
        full_wuk = torch.concatenate((torch.flip(half_wuk,[0])[:-1,:], half_wuk), 0)
        return full_wuk

    def input_encoding(self, input, encode='rate'):
        img_data = input / max(input.reshape(-1))
        if encode == 'rate':
            spikes = spikegen.rate(img_data, num_steps=self.time_window)
        elif encode == 'latency':
            spikes = spikegen.rate(img_data, num_steps=self.time_window)
        ftracekernel = np.array([[0.8 ** i] for i in range(20)])
        ftracekernel = np.concatenate((np.zeros((19, 1)), ftracekernel))
        btracekernel = np.flip(ftracekernel)
        ftrace = signal.convolve2d(spikes.detach().numpy(), btracekernel, 'same')
        btrace = signal.convolve2d(spikes.detach().numpy(), ftracekernel, 'same')
        return spikes, ftrace, btrace

    def encoding_layer(self, img):
        inhibitory_psp = np.array((0.5, 0.2789, 0.0458)) * 0.5
        img = (img / max(img.reshape(-1))) * self.V_th / 2  # 最快的神经元也需要2回合发放
        V_tn = np.array([img for i in range(self.time_window)])  # index: 0=timestep, 1=neuron
        U_tn = torch.zeros((self.time_window, self.in_number))
        O_tn = torch.zeros(U_tn.shape)
        for t in range(1, self.time_window):
            U_tn[t] = torch.mul((U_tn[t - 1, ...] + V_tn[t]), (1. - O_tn[t - 1, ...]))
            O_tn[t] = copy.deepcopy(U_tn[t])
            O_tn[t][O_tn[t] >= self.V_th] = 1
            O_tn[t][O_tn[t] < self.V_th] = 0
            V_tnd = np.outer(inhibitory_psp, np.ones(self.in_number))
            V_tn[t + 1:t + 1 + len(inhibitory_psp), :]


    def LIF_layer2(self, syns, ftrace, btrace):
        U_tn = torch.zeros((self.time_window, self.W.shape[1], self.W.shape[2]))
        O_tn = torch.zeros((self.time_window, self.W.shape[1], self.W.shape[2]))
        W_2d = self.W.reshape(self.in_number, self.output.shape[0] * self.output.shape[1])
        V_tn1 = torch.mm(syns, torch.from_numpy(W_2d))
        V_tn = torch.reshape(V_tn1, (self.time_window, self.output.shape[0], self.output.shape[1]))
        # value_lateral = torch.zeros((self.time_window, self.W.shape[1], self.W.shape[2]))
        # kernel = create_DOG_kernel(size=128, sigma1=3, sigma2=5) * 100
        fired = False
        V_th_mul = 1
        while fired == False:
            for t in range(1, self.time_window):
                # value_lateral = signal.convolve2d(O_tn[t-1, ...], kernel, mode='same', boundary='wrap')
                U_tn[t, ...] = torch.mul((self.tau_layer2 * U_tn[t - 1, ...] + V_tn[t, ...]),
                                         (1. - O_tn[t - 1, ...])) + self.V_rest * O_tn[t - 1, ...]  # + value_lateral
                # U_tn[t][sum(O_tn[max(0,t-self.RFP):t+1])>1e-3]=self.V_rest # 若处在不应期，膜电位=V_rest np.sum(O_tn[max(0,t-self.RFP):t-1, ...])
                U_data2 = copy.deepcopy(U_tn[t, ...])
                U_data2[U_data2 < self.V_th * V_th_mul] = 0
                U_data2[U_data2 >= self.V_th * V_th_mul] = 1
                O_tn[t, ...] = copy.deepcopy(U_data2)
                if max(O_tn[t].reshape(-1)) > .5:
                    winner_index = np.unravel_index(np.argmax(U_tn[t]), U_data2.shape)
                    winner_t = t
                    ft = ftrace[t]
                    bt = btrace[t]
                    fired = True
                    break
                if t == self.time_window - 1:
                    V_th_mul *= 0.9
                    # print(f'recalculating, threshold multiplier={V_th_mul}')
        return O_tn, winner_t, winner_index, ft, bt


    def update_w(self, tau_LTP, tau_LTD, weight_updtknl, t_winner, t_post_index, ft, bt, eta):
        delta_w = np.zeros((self.in_number, self.output.shape[0], self.output.shape[1]))
        delta_w_i = ft - bt
        xp1 = t_post_index[0]
        xp2 = t_post_index[1]
        X1 = self.output.shape[0]
        X2 = self.output.shape[1]
        delta_w = np.einsum('i,jk->ijk', eta * delta_w_i,
                            weight_updtknl[X1 - xp1 - 1:2 * X1 - xp1 - 1, X2 - xp2 - 1:2 * X2 - xp2 - 1] )
        # delta_wbool=np.zeros(np.shape(delta_w))
        # delta_wbool[delta_w>0]=1
        # mu=3
        # delta_w = 2**(2*mu)*np.multiply((np.multiply(delta_wbool,np.power(1-self.W, mu)) + np.multiply((1-delta_wbool),np.power(self.W, mu))) , delta_w)
        #print(round(max(delta_w.reshape(-1)),3), round(min(delta_w.reshape(-1)),3), round(sum(delta_w.reshape(-1)),3))
        self.W = np.add(self.W, delta_w)
        self.W = norm_W(self.W)
        return self.W  # self.t_post, self.t_post_index


    def forward(self, input, tau_LTP, tau_LTD, eta, weight_updtknl):
        img_data = self.input_norm_const * input / torch.sqrt(torch.sum(torch.square(input)))
        input_spike, ftrace, btrace = SOM_STDP.input_encoding(self, img_data)
        #print(f'{input_spike.reshape(-1).sum()} spikes fired in layer 1')
        syns = psp(input_spike) * self.V_peak
        O_tn, winner_t, winner_index, ft, bt = SOM_STDP.LIF_layer2(self, syns, ftrace, btrace)
        SOM_STDP.update_w(self, tau_LTP, tau_LTD, weight_updtknl, winner_t, winner_index, ft, bt, eta)
        # print(self.W)
        # self.W, t_post, t_post_index = self.update_w(self, 15, 0.5, -0.5, 0.5)
        return O_tn, self.W, winner_t, winner_index  # self.t_post, self.t_post_index



