
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator():
    def __init__(
        self, 
        mode, 
        nms_iou             = 0.7,
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 600,
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 300,
        min_size            = 16
    
    ):
        #-----------------------------------#
        #   设置预测还是训练
        #-----------------------------------#
        self.mode               = mode
        #-----------------------------------#
        #   建议框非极大抑制的iou大小
        #-----------------------------------#
        self.nms_iou            = nms_iou
        #-----------------------------------#
        #   训练用到的建议框数量
        #-----------------------------------#
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms
        #-----------------------------------#
        #   预测用到的建议框数量
        #-----------------------------------#
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms
        self.min_size           = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else:
            n_pre_nms   = self.n_test_pre_nms
            n_post_nms  = self.n_test_post_nms

        #-----------------------------------#
        #   将先验框转换成tensor
        #-----------------------------------#
        anchor = torch.from_numpy(anchor).type_as(loc)
        #-----------------------------------#
        #   将RPN网络预测结果转化成建议框
        #-----------------------------------#
        roi = loc2bbox(anchor, loc)
        #-----------------------------------#
        #   防止建议框超出图像边缘
        #-----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        #-----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        #-----------------------------------#
        min_size    = self.min_size * scale
        keep        = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        #-----------------------------------#
        #   将对应的建议框保留下来
        #-----------------------------------#
        roi         = roi[keep, :]
        score       = score[keep]

        #-----------------------------------#
        #   根据得分进行排序，取出建议框
        #-----------------------------------#
        order       = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order   = order[:n_pre_nms]
        roi     = roi[order, :]
        score   = score[order]

        #-----------------------------------#
        #   对建议框进行非极大抑制
        #   使用官方的非极大抑制会快非常多
        #-----------------------------------#
        keep    = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep        = torch.cat([keep, keep[index_extra]])
        keep    = keep[:n_post_nms]
        roi     = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        in_channels     = 512, 
        mid_channels    = 512, 
        ratios          = [0.5, 1, 2],
        anchor_scales   = [8, 16, 32], 
        feat_stride     = 16,
        mode            = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base    = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor            = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        self.conv1  = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        self.score  = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        self.loc    = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        #-----------------------------------------#
        #   特征点间距步长
        #-----------------------------------------#
        self.feat_stride    = feat_stride
        #-----------------------------------------#
        #   用于对建议框解码并进行非极大抑制
        #-----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)
        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        x = F.relu(self.conv1(x))
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        rpn_locs = self.loc(x)
        # 在这里，输出的rpn_locs通道 每四个通道为在这个点上的一个锚框的四个偏移量
        # 首先先把通道数放最后，会变成h*w*channel的维度，假设h w 都为2，channel为8，代表两个锚框，如下
# tensor([[[[0.1000, 0.2000], 
#           [0.3000, 0.4000]],

#          [[0.5000, 0.6000], 
#           [0.7000, 0.8000]],

#          [[0.9000, 1.0000], 
#           [1.1000, 1.2000]],

#          [[1.3000, 1.4000],
#           [1.5000, 1.6000]],

#          [[1.7000, 1.8000],
#           [1.9000, 2.0000]],

#          [[2.1000, 2.2000],
#           [2.3000, 2.4000]],

#          [[2.5000, 2.6000],
#           [2.7000, 2.8000]],

#          [[2.9000, 3.0000],
#           [3.1000, 3.2000]]]])

# Permuted Tensor:
# tensor([[[[0.1000, 0.5000, 0.9000, 1.3000, 1.7000, 2.1000, 2.5000, 2.9000],
#           [0.2000, 0.6000, 1.0000, 1.4000, 1.8000, 2.2000, 2.6000, 3.0000]],

#          [[0.3000, 0.7000, 1.1000, 1.5000, 1.9000, 2.3000, 2.7000, 3.1000],
#           [0.4000, 0.8000, 1.2000, 1.6000, 2.0000, 2.4000, 2.8000, 3.2000]]]])
        # 那么，其实这样就三个维度，第一个维度是深度，代表这特征图上的h，第二个维度是每个深度有多少行，代表的是特征图上的w
        # 第三个维度就是每个深度的某一行有多少列了，代表的是特征图上的channel
        # 比如在第一个深度的第一行，就代表特征图上第一个h和第一个w所表示的点的两个锚框的八个偏移值
        # 例如，第一行代表特征图上的第一行第一列两个锚框的八个偏移值
        # 第二行代表特征图上第一行第二列两个锚框的八个偏移值。。 
        # 第n行代表特征图上第 n // 列数 行，n - n//列数 列 的八个偏移值 这里不太清楚算的是否对，一般不用这个
        # 在 permute 后的张量中，你可以直接通过 [batch_index, row_index, col_index, :] 来访问特定位置的锚点信息。
        # 第n行第m列的定位直接就是 [batch_index, row_index, col_index, :]

        # reshape之后，可以看到，就相当于把那八个偏移值又分成了两行，
        # 第一行代表特征图上的第一行第一列第一个锚框的四个偏移值
        # 第二行代表特征图上的第一行第一列第二个锚框的四个偏移值
        # 第三行代表特征图上的第一行第二列第一个锚框的四个偏移值 。。 以此类推
        # 特征图上第n行m列，第a个锚框的点定位， 
        # i = n * W * A + m * A + a

#         Reshaped Tensor:
# tensor([[[0.1000, 0.5000, 0.9000, 1.3000],
#          [1.7000, 2.1000, 2.5000, 2.9000],
#          [0.2000, 0.6000, 1.0000, 1.4000],
#          [1.8000, 2.2000, 2.6000, 3.0000],
#          [0.3000, 0.7000, 1.1000, 1.5000],
#          [1.9000, 2.3000, 2.7000, 3.1000],
#          [0.4000, 0.8000, 1.2000, 1.6000],
#          [2.0000, 2.4000, 2.8000, 3.2000]]])
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  #　（batch_size, h * w * num_anchors, 4)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        
        #--------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores       = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores       = rpn_fg_scores.view(n, -1)

        #------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        #------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        rois        = list()
        roi_indices = list()
        for i in range(n):
            roi         = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois        = torch.cat(rois, dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        
  
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
