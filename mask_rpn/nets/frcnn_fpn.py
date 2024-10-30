import torch.nn as nn

from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet50_FPN import resnet50_FPN
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16
from nets.fpn_mask_rpn import Mask_RPN
import torch
from .resnet50_FPN import get_FPN_features
from spikingjelly.activation_based import functional
def adjust_and_map_rois(rois, S):
    # 为了保持数据在同一设备上，我们将使用 rois 直接进行计算
    # 计算调整后的坐标
    xmin_adj = rois[:, 0] 
    xmax_adj = rois[:, 2] 
    ymin_adj = rois[:, 1]
    ymax_adj = rois[:, 3]


    # 将调整后的坐标映射回原始尺寸
    xmin_orig = xmin_adj * S
    xmax_orig = xmax_adj * S
    ymin_orig = ymin_adj * S
    ymax_orig = ymax_adj * S

    # 返回调整后的 ROI，这些 ROI 现在映射回原始图像尺寸
    adjusted_rois = torch.stack([xmin_orig, ymin_orig, xmax_orig, ymax_orig], dim=1)
    return adjusted_rois

class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,  
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.get_fpn_features = get_FPN_features([256, 512, 1024]).cuda()
        #---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        #---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            #---------------------------------#
            #   构建建议框网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = VGG16RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 7,
                spatial_scale   = 1,
                classifier      = classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50_FPN(pretrained)
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            # self.mask_rpn1 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.068)
            # self.mask_rpn2 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.043)
            # self.mask_rpn3 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.033)

            # self.mask_rpn1 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.07)
            # self.mask_rpn2 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.043)
            # self.mask_rpn3 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.034)
            
            self.mask_rpn1 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.09)
            self.mask_rpn2 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.1)
            self.mask_rpn3 = Mask_RPN(num_classes=num_classes, use_figure_layers=True, v_th=0.08)
            #---------------------------------#
            #   构建classifier网络
            #---------------------------------#
            self.head = Resnet50RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )
            
    def forward(self, x, scale=1., count=1,mode="forward"):
        # print(mode)
         
        if mode == "forward":
            #---------------------------------#
            #   计算输入图片的大小
            n           = x.shape[0]

            #---------------------------------#
            img_size        = x.shape[2:]
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature1, base_feature2, base_feature3    = self.extractor.forward(x)
            feature1, feature2, feature3 = self.get_fpn_features.forward(base_feature1, base_feature2, base_feature3)

            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            functional.reset_net(self.mask_rpn1)
            mask_rpn1, rois1, box_counts1 = self.mask_rpn1(feature1)
            functional.reset_net(self.mask_rpn2)
            mask_rpn2, rois2, box_counts2 = self.mask_rpn2(feature2)
            functional.reset_net(self.mask_rpn3)
            mask_rpn3, rois3, box_counts3 = self.mask_rpn3(feature3)
            roi_indice = []
            rois_tensor = []
            for i in range(n):
                roi_indice.append(torch.full((box_counts1[i],), i, dtype=torch.long))
                roi_indice.append(torch.full((box_counts2[i],), i, dtype=torch.long))
                roi_indice.append(torch.full((box_counts3[i],), i, dtype=torch.long))
                
                roi = torch.tensor(rois1[i], dtype=torch.float)
                roi = adjust_and_map_rois(roi, 2)
                rois_tensor.append(roi)

                roi = torch.tensor(rois2[i], dtype=torch.float)

                roi = adjust_and_map_rois(roi, 4)
                rois_tensor.append(roi)

                roi = torch.tensor(rois3[i], dtype=torch.float)
                roi = adjust_and_map_rois(roi, 8)
                rois_tensor.append(roi)

            # _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, img_size, scale)
            roi_indices  = torch.cat(roi_indice, dim=0).to(self.device)
            rois_tensors     = torch.cat(rois_tensor, dim=0).to(self.device)
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head.forward(base_feature3, rois_tensors, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois_tensors, roi_indices
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature1, base_feature2, base_feature3    = self.extractor.forward(x)
            return base_feature1, base_feature2, base_feature3
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            #rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            if count == 1:
                functional.reset_net(self.mask_rpn1)
                mask_rpn,rois, box_count = self.mask_rpn1(base_feature)
            if count == 2:
                functional.reset_net(self.mask_rpn2)
                mask_rpn,rois, box_count = self.mask_rpn2(base_feature)
            if count == 3:
                functional.reset_net(self.mask_rpn3)
                mask_rpn,rois, box_count = self.mask_rpn3(base_feature)
            # print(rois.shape)
    
            return mask_rpn, rois, box_count
            #return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            # print(rois.shape)
            roi_cls_locs, roi_scores    = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
