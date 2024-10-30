
import cv2
from spikingjelly.activation_based import neuron, surrogate, layer
import torch.nn as nn
import numpy as np



def find_connected_components(x, T):
    batch_size, num_classes, height, width = x.size()
    all_boxes = []  # 存储所有batch的boxes
    box_counts = []  # 存储每个batch中的候选框数量
    for b in range(batch_size):
        batch_boxes = []  # 存储当前batch的所有boxes
        for c in range(num_classes):
            mask = (x[b, c].detach().cpu().numpy() == T).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for label in range(1, num_labels):
                x_min, y_min, w, h, area = stats[label]
                x_max = x_min + w - 1
                y_max = y_min + h - 1
                if x_max - x_min <= 2 or y_max-y_min <= 2:
                    continue
                batch_boxes.append([x_min, y_min, x_max, y_max])

        if len(batch_boxes) == 0:
            batch_boxes.append([0, 0, height, width])

        all_boxes.append(batch_boxes)
        box_counts.append(len(batch_boxes)) # 添加当前batch的候选框数量

    return all_boxes,  box_counts

# 其他代码保持不变


class Mask_RPN(nn.Module):
    def __init__(self, num_classes, use_figure_layers, v_th):
        super(Mask_RPN, self).__init__()
        self.tau = 2.0
        v_th = v_th
        if use_figure_layers:
            self.figure_layers = 5
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.sn1 = neuron.LIFNode(tau=self.tau, surrogate_function=surrogate.ATan(), step_mode='m', v_threshold=v_th)
        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.sn2 = neuron.LIFNode(tau=self.tau, surrogate_function=surrogate.ATan(), step_mode='m', v_threshold=v_th)
        self.conv3 = layer.SeqToANNContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.sn3 = neuron.LIFNode(tau=self.tau, surrogate_function=surrogate.ATan(), step_mode='m', v_threshold=v_th)
        self.conv4 = layer.SeqToANNContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.sn4 = neuron.LIFNode(tau=self.tau, surrogate_function=surrogate.ATan(), step_mode='m', v_threshold=v_th)

        self.deconv = layer.SeqToANNContainer(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        self.sn5 = neuron.LIFNode(tau=self.tau, surrogate_function=surrogate.ATan(), step_mode='m', v_threshold=v_th)
        if use_figure_layers:
            self.conv6 = layer.SeqToANNContainer(nn.Conv2d(256, num_classes*self.figure_layers, kernel_size=1, padding=0))
        else:
            self.conv6 = layer.SeqToANNContainer(nn.Conv2d(256, num_classes, kernel_size=1, padding=0))
        self.sn6 = neuron.LIFNode(tau=self.tau, surrogate_function=surrogate.ATan(), step_mode='m', v_threshold=v_th)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = x.unsqueeze(0)
        x = x.repeat(4, 1, 1, 1, 1)
        x = self.sn1(self.conv1(x))
        x = self.sn2(self.conv2(x))
        x = self.sn3(self.conv3(x))
        x = self.sn4(self.conv4(x))
        x = self.sn5(self.deconv(x))
        x = self.sn6(self.conv6(x))
        #　600*600的图像到这里是(T, batchsize, numclasses, 76, 76)


        x_sum = x.sum(0)
        # print(x_sum.shape)
        # print(x_sum.mean())

        rois,  box_count = find_connected_components(x_sum, 4)
 


        return x_sum, rois, box_count


