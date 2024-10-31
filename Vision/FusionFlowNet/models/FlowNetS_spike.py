import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_
from .util import predict_flow, crop_like, conv_s, conv, deconv
__all__ = ['fusion_flownets']


class SpikingNN(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        out = input.gt(1e-5).type(torch.cuda.FloatTensor)
        out -= input.lt(-1e-5).type(torch.cuda.FloatTensor)
        return out

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() <= 1e-5] = 0
        grad_input[input.gt(1e-5)] = 1/threshold_k
        grad_input[input.lt(-1e-5)] = 1/threshold_neg_k
        return grad_input

def IF_Neuron(membrane_potential, threshold, threshold_neg):
    global threshold_k, threshold_neg_k
    threshold_k = threshold
    threshold_neg_k = threshold_neg
    # check exceed membrane potential and reset
    ex_membrane_1 = nn.functional.threshold(membrane_potential, threshold_k, 0)
    ex_membrane_2 = -nn.functional.threshold(-membrane_potential, threshold_neg_k, 0)
    ex_membrane = ex_membrane_1 + ex_membrane_2
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN.apply(ex_membrane)
    out = out.detach() + (1/threshold)*out.gt(1e-5) + (1/threshold_neg)*out.lt(-1e-5) - (1/threshold)*out.detach().gt(1e-5) - (1/threshold_neg)*out.detach().lt(-1e-5)

    return membrane_potential, out

# class SpikingNN(torch.autograd.Function):
#     def forward(self, input):
#         self.save_for_backward(input)
#         return input.gt(1e-5).type(torch.cuda.FloatTensor)

#     def backward(self, grad_output):
#         input, = self.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input <= 1e-5] = 0
#         return grad_input

# def IF_Neuron(membrane_potential, threshold, threshold_neg):
#     global threshold_k
#     threshold_k = threshold
#     # check exceed membrane potential and reset
#     ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
#     membrane_potential = membrane_potential - ex_membrane # hard reset
#     # generate spike
#     out = SpikingNN.apply(ex_membrane)
#     out = out.detach() + (1/threshold)*out - (1/threshold)*out.detach()

#     return membrane_potential, out



class FlowNetS_spike(nn.Module):
    expansion = 1
    def __init__(self,batchNorm=True):
        super(FlowNetS_spike,self).__init__()
        self.batchNorm = batchNorm

        self.sconv1   = conv_s(self.batchNorm,   4,   32, kernel_size=3, stride=2)
        self.sconv2   = conv_s(self.batchNorm,  32,   64, kernel_size=3, stride=2)
        self.sconv3   = conv_s(self.batchNorm,  64,  128, kernel_size=3, stride=2)
        self.sconv4   = conv_s(self.batchNorm, 128,  256, kernel_size=3, stride=2)

        self.aconv1   = conv(self.batchNorm,   2,   32, kernel_size=3, stride=2)
        self.aconv2   = conv(self.batchNorm,  32,   64, kernel_size=3, stride=2)
        self.aconv3   = conv(self.batchNorm,  64,  128, kernel_size=3, stride=2)
        self.aconv4   = conv(self.batchNorm, 128,  256, kernel_size=3, stride=2)

        self.conv_r11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        self.deconv3  = deconv(self.batchNorm, 512,128)
        self.deconv2  = deconv(self.batchNorm, 384+2,64)
        self.deconv1  = deconv(self.batchNorm, 192+2,32)

        self.predict_flow4 = predict_flow(self.batchNorm, 32)
        self.predict_flow3 = predict_flow(self.batchNorm, 32)
        self.predict_flow2 = predict_flow(self.batchNorm, 32)
        self.predict_flow1 = predict_flow(self.batchNorm, 32)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(in_channels=384+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(in_channels=192+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(in_channels=96+2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, sinput, ainput, image_resize, sp_threshold, sp_threshold_neg):
        threshold = sp_threshold
        threshold_neg = sp_threshold_neg

        smem_1 = torch.zeros(sinput.size(0), 32, int(image_resize/2), int(image_resize/2)).cuda()
        smem_2 = torch.zeros(sinput.size(0), 64, int(image_resize/4), int(image_resize/4)).cuda()
        smem_3 = torch.zeros(sinput.size(0), 128, int(image_resize/8), int(image_resize/8)).cuda()
        smem_4 = torch.zeros(sinput.size(0), 256, int(image_resize/16), int(image_resize/16)).cuda()

        smem_1_total = torch.zeros(sinput.size(0), 32, int(image_resize/2), int(image_resize/2)).cuda()
        smem_2_total = torch.zeros(sinput.size(0), 64, int(image_resize/4), int(image_resize/4)).cuda()
        smem_3_total = torch.zeros(sinput.size(0), 128, int(image_resize/8), int(image_resize/8)).cuda()
        smem_4_total = torch.zeros(sinput.size(0), 256, int(image_resize/16), int(image_resize/16)).cuda()

        # print("sinput:", sinput.size(), "ainput:", ainput.size())
        # sinput: torch.Size([8, 4, 256, 256, 5]) ainput: torch.Size([8, 2, 256, 256, 1])

        for i in range(sinput.size(4)):
            input11 = sinput[:, :, :, :, i].cuda()
            # print("input11:", input11.size())
            # input11: torch.Size([2, 4, 256, 256])

            current_1 = self.sconv1(input11)
            smem_1 = smem_1 + current_1
            smem_1_total = smem_1_total + current_1
            smem_1, out_sconv1 = IF_Neuron(smem_1, threshold, threshold_neg)
            # smem_1_total = smem_1_total + out_sconv1

            current_2 = self.sconv2(out_sconv1)
            smem_2 = smem_2 + current_2
            smem_2_total = smem_2_total + current_2
            smem_2, out_sconv2 = IF_Neuron(smem_2, threshold, threshold_neg)
            # smem_2_total = smem_2_total + out_sconv2

            current_3 = self.sconv3(out_sconv2)
            smem_3 = smem_3 + current_3
            smem_3_total = smem_3_total + current_3
            smem_3, out_sconv3 = IF_Neuron(smem_3, threshold, threshold_neg)
            # smem_3_total = smem_3_total + out_sconv3

            current_4 = self.sconv4(out_sconv3)
            smem_4 = smem_4 + current_4
            smem_4_total = smem_4_total + current_4
            smem_4, out_sconv4 = IF_Neuron(smem_4, threshold, threshold_neg)
            # smem_4_total = smem_4_total + out_sconv4

        smem_4_residual = 0
        smem_3_residual = 0
        smem_2_residual = 0

        out_sconv4 = smem_4_total + smem_4_residual
        out_sconv3 = smem_3_total + smem_3_residual
        out_sconv2 = smem_2_total + smem_2_residual
        out_sconv1 = smem_1_total

        amem_1 = torch.zeros(ainput.size(0), 32, int(image_resize/2), int(image_resize/2)).cuda()
        amem_2 = torch.zeros(ainput.size(0), 64, int(image_resize/4), int(image_resize/4)).cuda()
        amem_3 = torch.zeros(ainput.size(0), 128, int(image_resize/8), int(image_resize/8)).cuda()
        amem_4 = torch.zeros(ainput.size(0), 256, int(image_resize/16), int(image_resize/16)).cuda()

        for i in range(ainput.size(4)):
            input22 = ainput[:, :, :, :, i].cuda()
            # print("input22:", input22.size())
            # input22: torch.Size([2, 2, 256, 256])

            amem_1 = self.aconv1(input22)
            out_aconv1 = amem_1

            amem_2 = self.aconv2(out_aconv1)
            out_aconv2 = amem_2

            amem_3 = self.aconv3(out_aconv2)
            out_aconv3 = amem_3

            amem_4 = self.aconv4(out_aconv3)
            out_aconv4 = amem_4

        out_conv = torch.cat((out_sconv4, out_aconv4), 1)
        # print("os4:", out_sconv4.size(), "oa4:", out_aconv4.size())
        # os4: torch.Size([2, 256, 16, 16]) oa4: torch.Size([2, 256, 16, 16])

        out_rconv11 = self.conv_r11(out_conv)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

        flow4 = self.predict_flow4(self.upsampled_flow4_to_3(out_rconv22))
        # out_rconv22: torch.Size([8, 512, 16, 16])
        # print("out_rconv22:", out_rconv22.size())
        # print("flow4:",flow4.size(), "out_sconv3:",out_sconv3.size())
        flow4_up = crop_like(flow4, out_sconv3)
        # print("out_rconv22:", out_rconv22.size(), "out_sconv3:",out_sconv3.size())
        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_sconv3)

        concat3 = torch.cat((out_sconv3, out_aconv3,out_deconv3,flow4_up),1)
        # print("concat3:", concat3.size())
        # print("out_sconv3:", out_sconv3.size(), "out_aconv3:", out_aconv3.size(), "out_deconv3:", out_deconv3.size(), "flow4_up:", flow4_up.size())
        # concat3: torch.Size([8, 386, 32, 32])
        # out_sconv3: torch.Size([8, 128, 32, 32]) out_aconv3: torch.Size([8, 128, 32, 32]) out_deconv3: torch.Size([8, 128, 32, 32]) flow4_up: torch.Size([8, 2, 32, 32])
        # print("out_sconv3:", out_sconv3, "out_aconv3:", out_aconv3, "out_deconv3:", out_deconv3, "flow4_up:", flow4_up)
        flow3 = self.predict_flow3(self.upsampled_flow3_to_2(concat3))
        # flow3: torch.Size([8, 2, 64, 64])
        # print("flow3:", flow3.size(), "out_sconv2:", out_sconv2.size())
        flow3_up = crop_like(flow3, out_sconv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_sconv2)

        concat2 = torch.cat((out_sconv2, out_aconv2,out_deconv2,flow3_up),1)
        # print("concat2:", concat2.size())
        # concat2: torch.Size([8, 194, 64, 64])
        # out_sconv2: torch.Size([8, 64, 64, 64]) out_aconv2: torch.Size([8, 64, 64, 64]) out_deconv2: torch.Size([8, 64, 64, 64]) flow3_up: torch.Size([8, 2, 64, 64])
        # print("out_sconv2:", out_sconv2.size(), "out_aconv2:", out_aconv2.size(), "out_deconv2:", out_deconv2.size(), "flow3_up:", flow3_up.size())
        flow2 = self.predict_flow2(self.upsampled_flow2_to_1(concat2))
        # flow2: torch.Size([8, 2, 128, 128])
        # print("flow2:", flow2.size(), "out_sconv1:", out_sconv1.size())
        flow2_up = crop_like(flow2, out_sconv1)
        out_deconv1 = crop_like(self.deconv1(concat2), out_sconv1)

        concat1 = torch.cat((out_sconv1, out_aconv1,out_deconv1,flow2_up),1)
        # print("concat1:", concat1.size())
        # concat1: torch.Size([8, 98, 128, 128])
        # out_sconv1: torch.Size([8, 32, 128, 128]) out_aconv1: torch.Size([8, 32, 128, 128]) out_deconv1: torch.Size([8, 32, 128, 128]) flow2_up: torch.Size([8, 2, 128, 128])
        # print("out_sconv1:", out_sconv1.size(), "out_aconv1:", out_aconv1.size(), "out_deconv1:", out_deconv1.size(), "flow2_up:", flow2_up.size())
        flow1 = self.predict_flow1(self.upsampled_flow1_to_0(concat1))
        # flow1: torch.Size([8, 2, 256, 256])
        # print("flow1:", flow1.size())

        if self.training:
            return flow1,flow2,flow3,flow4
        else:
            return flow1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def fusion_flownets(data=None):
    model = FlowNetS_spike(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

