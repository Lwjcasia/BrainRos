import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import scipy.ndimage
from .resnet50_FPN import get_FPN_features
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
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

def bbox_mapping(mask, bboxes, labels, S, T, figure_layers, min_area, max_area):
    for bbox, label in zip(bboxes, labels):


        xmin, ymin, xmax, ymax = bbox
        bbox_area = (xmax - xmin) * (ymax - ymin)
        # print(bbox_area)
        # # 检查面积是否在指定范围内
        # if bbox_area <= min_area or bbox_area > max_area:
        #     continue  # 如果不在范围内，跳过当前边界框
        # 调整坐标
        xmin_adj = int(xmin / S) + 1
        xmax_adj = int(np.ceil(xmax / S)) - 1
        ymin_adj = int(ymin / S) + 1 
        ymax_adj = int(np.ceil(ymax / S)) - 1

        xmin_adj = min(xmin_adj, xmax_adj)
        xmax_adj = max(xmin_adj, xmax_adj)
        ymin_adj = min(ymin_adj, ymax_adj)
        ymax_adj = max(ymin_adj, ymax_adj)
        
        if isinstance(figure_layers, int):       
            layer_start = int(label) * figure_layers
            layer_end = layer_start + figure_layers
            # 测试不按类别


            for layer in range(layer_start, layer_end):
            # 在属于类别的通道内循环尝试标记
                # 检查当前区域是否已被标记为T
                if np.any(mask[layer, ymin_adj:ymax_adj + 1, xmin_adj:xmax_adj + 1] == T):
                    continue  # 如果当前通道已有标记，则尝试下一个通道
                # 标记当前通道的对应区域
                mask[layer, ymin_adj:ymax_adj + 1, xmin_adj:xmax_adj + 1] = T
                break  # 成功标记后，跳出循环，不再标记其他通道
        # 标记对应类别的通道
        else:
            mask[int(label), ymin_adj:ymax_adj + 1, xmin_adj:xmax_adj + 1] = T

    return mask

def create_foreground_weight_map(mask_label, T, max_weight=1, min_weight=0.1):
    # 确保mask_label是PyTorch Tensor
    if not isinstance(mask_label, torch.Tensor):
        mask_label = torch.tensor(mask_label, dtype=torch.uint8)

    # 计算前景区域
    foregrounds = (mask_label == T)

    padded_foregrounds = []
    distances = []
    # 遍历每个channel的前景
    for foreground in foregrounds:
        foreground_np = foreground.cpu().numpy().astype(np.uint8)
        # 如果全是前景，增加一圈0（背景）
        if np.all(foreground_np == 1):
            padded_foreground = np.pad(foreground_np, pad_width=1, mode='constant', constant_values=0)

        else:
            padded_foreground = foreground_np
        padded_foregrounds.append(padded_foreground)

        # 计算距离变换
        distance = scipy.ndimage.distance_transform_edt(padded_foreground == 1) - 1
        if np.all(foreground_np == 1):
            distance = distance[1:-1, 1:-1]
        distances.append(distance)

    weight_maps = []
    for idx, distance in enumerate(distances):
        # 归一化距离：转换为权重，使权重在min_weight和max_weight之间线性变化
        max_dist = np.max(distance)
        if max_dist > 0:
            weight_map = max_weight - (max_weight - min_weight) * distance / max_dist
        else:
            weight_map = np.full_like(distance, max_weight)

        # 转换为Tensor并为背景设置权重0
        weight_map = torch.from_numpy(weight_map).float()
        weight_map[foregrounds[idx] == 0] = 0
        weight_maps.append(weight_map)

    return torch.stack(weight_maps)




def create_background_weight_map(mask_label, T, max_weight=1, min_weight=0.1):
    # 确保mask_label是PyTorch Tensor
    if not isinstance(mask_label, torch.Tensor):
        mask_label = torch.tensor(mask_label, dtype=torch.uint8)
    
    # 计算背景区域
    background = (mask_label == 0)

    weight_maps = []
    for i in range(background.shape[0]):  # 遍历每个通道
        foreground_np = background[i].cpu().numpy().astype(np.uint8)

        # 距离变换：计算每个背景点到最近前景点的距离
        distances = scipy.ndimage.distance_transform_edt(foreground_np == 1)

        # 检查是否全是背景
        if np.all(foreground_np == 1):
            # 全为背景，设定权重为最小权重
            weight_map = np.full_like(distances, 0)
        else:
            # 归一化距离：转换为权重，使权重在min_weight和max_weight之间线性变化
            max_dist = np.max(distances)
            if max_dist > 0:
                # 归一化距离，并线性映射到权重范围
                weight_map = min_weight + (max_weight - min_weight) * distances / max_dist
            else:
                # 所有背景点都在前景边界上（不太可能，但以防万一）
                weight_map = np.full_like(distances, min_weight)

        # 为前景设置权重为0
        weight_map[foreground_np == 0] = 0
        weight_maps.append(weight_map)

    # 将权重图列表转换为Tensor
    weight_maps_tensor = torch.from_numpy(np.stack(weight_maps)).float()
    return weight_maps_tensor

class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample       = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio      = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        #----------------------------------------------#
        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        #----------------------------------------------#
        ious = bbox_iou(anchor, bbox)

        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        #---------------------------------------------------------#
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        #---------------------------------------------------------#
        argmax_ious = ious.argmax(axis=1)
        #---------------------------------------------------------#
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        #---------------------------------------------------------#
        max_ious = np.max(ious, axis=1)
        #---------------------------------------------------------#
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        #---------------------------------------------------------#
        gt_argmax_ious = ious.argmax(axis=0)
        #---------------------------------------------------------#
        #   保证每一个真实框都存在对应的先验框
        #---------------------------------------------------------#
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious
        
    def _create_label(self, anchor, bbox):
        # ------------------------------------------ #
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        # ------------------------------------------ #
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # ------------------------------------------------------------------------ #
        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        # ------------------------------------------------------------------------ #
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        
        # ----------------------------------------------------- #
        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        # ----------------------------------------------------- #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious)>0:
            label[gt_argmax_ious] = 1

        # ----------------------------------------------------- #
        #   判断正样本数量是否大于128，如果大于则限制在128
        # ----------------------------------------------------- #
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # ----------------------------------------------------- #
        #   平衡正负样本，保持总数量为256
        # ----------------------------------------------------- #
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label
    

class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.2, neg_iou_thresh_high=0.2, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # ----------------------------------------------------- #
        #   计算建议框和真实框的重合程度
        # ----------------------------------------------------- #
        iou = bbox_iou(roi, bbox)
        
        if len(bbox)==0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            #---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框  [num_roi, ]
            #---------------------------------------------------------#
            gt_assignment = iou.argmax(axis=1)
            #---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            #---------------------------------------------------------#
            max_iou = iou.max(axis=1)
            #---------------------------------------------------------#
            #   真实框的标签要+1因为有背景的存在
            #---------------------------------------------------------#
            gt_roi_label = label[gt_assignment] + 1

        #----------------------------------------------------------------#
        #   满足建议框和真实框重合程度大于pos_iou_thresh的作为正样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        #----------------------------------------------------------------#
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        #-----------------------------------------------------------------------------------------------------#
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        #-----------------------------------------------------------------------------------------------------#
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
            
        #---------------------------------------------------------#
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        #---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)
        # print(pos_index)
        # print(neg_index)
        # print(gt_assignment)
        # print(bbox)
        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        # print(sample_roi)
        # print(bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))
        # print(gt_roi_loc.shape)
        # print(gt_roi_loc)
        gt_roi_label = gt_roi_label[keep_index]
        # print(gt_roi_label.shape)
        # print(gt_roi_label)
        # gt_roi_label[pos_roi_per_this_image:] = 0
        gt_roi_label[len(pos_index):] = 0
        return sample_roi, gt_roi_loc, gt_roi_label
def ohem_loss(
    batch_size, cls_pred, cls_target, loc_pred, loc_target, smooth_l1_sigma=1.0
):
    smoothl1loss = nn.SmoothL1Loss(beta=smooth_l1_sigma, reduction='none')
    """
    Arguments:
        batch_size (int): number of sampled rois for bbox head training
        loc_pred (FloatTensor): [R, 4], location of positive rois
        loc_target (FloatTensor): [R, 4], location of positive rois
        pos_mask (FloatTensor): [R], binary mask for sampled positive rois
        cls_pred (FloatTensor): [R, C]
        cls_target (LongTensor): [R]
    Returns:
        cls_loss, loc_loss (FloatTensor)
    """

    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
    ohem_loc_loss = smoothl1loss(loc_pred, loc_target)
    ohem_loc_loss = ohem_loc_loss.sum(dim=1, keepdim=True)
    #这里先暂存下正常的分类loss和回归loss
    loss = ohem_cls_loss + ohem_loc_loss
    #然后对分类和回归loss求和
 
  
    sorted_ohem_loss, idx = torch.sort(loss, descending=True)
    #再对loss进行降序排列
    keep_num = min(sorted_ohem_loss.size()[0], batch_size)
    
    #得到需要保留的loss数量
    if keep_num < sorted_ohem_loss.size()[0]:
    #这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
        keep_idx_cuda = idx[:keep_num]
        #保留到需要keep的数目
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]
        #分类和回归保留相同的数目
    cls_loss = ohem_cls_loss.sum() / keep_num
    loc_loss = ohem_loc_loss.sum() / keep_num
    #然后分别对分类和回归loss求均值

    return cls_loss, loc_loss
class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model_train    = model_train
        self.optimizer      = optimizer

        self.rpn_sigma      = 1
        self.roi_sigma      = 1
        self.get_fpn_features = get_FPN_features([256, 512, 1024]).cuda()  # 三个层输出的通道数

        self.anchor_target_creator      = AnchorTargetCreator()
        self.proposal_target_creator    = ProposalTargetCreator()

        self.loc_normalize_std          = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        # print(pred_loc.shape)
        pred_loc    = pred_loc[gt_label > 0]  # 正样本进行训练
        # print(pred_loc.shape)
        gt_loc      = gt_loc[gt_label > 0]   # 同理

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
                regression_diff < (1. / sigma_squared),
                0.5 * sigma_squared * regression_diff ** 2,
                regression_diff - 0.5 / sigma_squared
            )
        regression_loss = regression_loss.sum()
        num_pos         = (gt_label > 0).sum().float()
        
        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss
    


    def forward(self, imgs, bboxes, labels, scale, batch):
        n           = imgs.shape[0]
        img_size    = imgs.shape[2:]
        #-------------------------------#
        #   获取公用特征层
        #-------------------------------#
        base_feature1, base_feature2, base_feature3 = self.model_train(imgs, mode = 'extractor')

        # print(base_feature3.shape)
        feature1, feature2, feature3 = self.get_fpn_features.forward(base_feature1, base_feature2, base_feature3)

        # print(feature3.shape)
        # print(feature1.shape)
        # -------------------------------------------------- #
        #   利用rpn网络获得调整参数、得分、建议框、先验框
        # -------------------------------------------------- #
        #rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(x = [base_feature, img_size], scale = scale, mode = 'rpn')
        # functional.reset_net(self.model_train)
        mask_rpns1, rois1, box_counts1 = self.model_train(x = [feature1, img_size], scale = scale, count=1, mode = 'rpn')  # 300*300
        # print(mask_rpns1.shape)
        # print('2')
        # functional.reset_net(self.model_train)
        mask_rpns2, rois2, box_counts2 = self.model_train(x = [feature2, img_size], scale = scale, count=2, mode = 'rpn')  # 150*150

        # # functional.reset_net(self.model_train)
        mask_rpns3, rois3, box_counts3 = self.model_train(x = [feature3, img_size], scale = scale, count=3, mode = 'rpn')  # 75*75



        # print(rois)
        mask_rpns_all = [mask_rpns1, mask_rpns2, mask_rpns3]
        rois_all = [rois1, rois2, rois3]
        box_counts_all = [box_counts1, box_counts2, box_counts3]

        # mask_rpns_all = [mask_rpns]
        # rois_all = [rois]
        # box_counts_all = [box_counts]

        rpn_losses, roi_loc_loss_all, roi_cls_loss_all  = 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels                 = [], [], [], []

        


        ##  先放 for i in range(n)  然后再是 mask_rpns这些循环，
        ## 每个i 里面的 mask_rpns这些循环里，需要把 
        # sample_roi，torch.ones(len(sample_roi)).type_as(mask_rpns) * roi_indice[0]，gt_roi_loc, gt_roi_label这些给拼起来
        
            
        for i in range(n):
            count = 0
            accumulated_sample_roi = []
            accumulated_ones = []
            accumulated_gt_roi_loc = []
            accumulated_gt_roi_label = []
            for mask_rpns, rois, box_counts in zip(mask_rpns_all, rois_all, box_counts_all):
                count += 1
                T = 4 
                img = imgs[i]
                bbox        = bboxes[i]
                label       = labels[i]
                # rpn_loc     = rpn_locs[i]
                # rpn_score   = rpn_scores[i]
                roi         = torch.tensor(rois[i]).type_as(mask_rpns)  # (n, 4)  # n是我们提出了多少个候选框，这个没固定
                if count == 1:
                    # print('count1')
                    # print(roi.shape)
                    roi = adjust_and_map_rois(roi, 2)    

                if count == 2:
                    # print('count2')
                    # print(roi.shape)
                    roi = adjust_and_map_rois(roi, 4) 

                if count == 3:  
                    # print('count3')
                    # print(roi.shape)
                    roi = adjust_and_map_rois(roi, 8) 

                # if batch == 1:
                #     if img.is_cuda:
                #         tensor_image = img.cpu()
                #     image_np = tensor_image.numpy()

                #     # 如果图像是CHW格式（通道，高度，宽度），则转换为HWC格式
                #     if image_np.shape[0] == 3:
                #         image_np = np.transpose(image_np, (1, 2, 0))

                #     # 将numpy数组转换为PIL图像
                #     image = Image.fromarray((image_np * 255).astype(np.uint8))
                #     draw = ImageDraw.Draw(image)
                #     # print(roi)
                #     # print(bbox)
                #     # 绘制每个ROI
                #     for roi_ in roi:
                #         x_min, y_min, x_max, y_max = roi_
                #         draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)

                #     # 检查文件是否存在，并生成不覆盖的新文件名
                #     file_index = 0
                #     while True:
                #         output_filename = f'mask_rpn_output_figurelayers_deloss1/output_image_with_rois_{batch}_{file_index}.png'
                #         if not os.path.exists(output_filename):
                #             image.save(output_filename)
                #             break
                #         file_index += 1

                mask_rpn = mask_rpns[i]  # numclasses*figure_layers, H/8, W/8
                # print(mask_rpn.shape)
                roi_indice = torch.full((box_counts[i],), i, dtype=torch.long)

                mask_label = np.zeros_like(mask_rpn.detach().cpu().numpy())

                # -------------------------------------------------- #
                # 根据bboxes内的坐标，将mask中对应的区域标记为1（或你可以选择其他非零值表示T）
                # mask作用为 为提取候选框用作标签
                if count == 1:
                    mask_label = bbox_mapping(mask_label, bbox, label, S=2, T=T, figure_layers=5, min_area=0, max_area=32**2)  # 第一个8是步长，到这的图像大小应该是/8，第二个是时间步长，目前也设置为4
                if count == 2:
                    mask_label = bbox_mapping(mask_label, bbox, label, S=4, T=T, figure_layers=5, min_area=32**2, max_area=96**2)
                if count == 3:
                    mask_label = bbox_mapping(mask_label, bbox, label, S=8, T=T, figure_layers=5, min_area=96**2, max_area=1e5**2) 
                # -------------------------------------------------- #
                mask_label = torch.tensor(mask_label).type_as(mask_rpn)

                # 创建前景和后景的布尔掩码
                foreground_mask = (mask_label == T)
                # print(foreground_mask.sum())
                background_mask = (mask_label == 0)

                use_weight = False

                if use_weight:
                    foreground_weight_map = create_foreground_weight_map(mask_label, T=T, max_weight=1, min_weight=0.1)
                    # background_weight_map = create_background_weight_map(mask_label, T=T, max_weight=1, min_weight=0.1)
                    #weight_map =torch.tensor(foreground_weight_map + background_weight_map).type_as(foreground_mask)
                    weight_map =torch.tensor(foreground_weight_map).type_as(foreground_mask)
                    
                    # 计算前景的MSE

                    if foreground_mask.any():
                        foreground_mse = ((mask_rpn[foreground_mask] - mask_label[foreground_mask]) ** 2 * weight_map[foreground_mask]).mean()
                    else:
                        foreground_mse = torch.tensor(0.0).to(mask_rpn.device)
                    # 计算后景的MSE
                    if background_mask.any():
                        background_mse = ((mask_rpn[background_mask] - mask_label[background_mask]) ** 2).mean()
                    else:
                        background_mse = torch.tensor(0.0).to(mask_rpn.device)
                else:
                    if foreground_mask.any():
                        foreground_mse = ((mask_rpn[foreground_mask] - mask_label[foreground_mask]) ** 2).mean()
                    else:
                        foreground_mse = torch.tensor(0.0).to(mask_rpn.device)
                    # 计算后景的MSE
                    if background_mask.any():
                        background_mse = ((mask_rpn[background_mask] - mask_label[background_mask]) ** 2).mean()
                    else:
                        background_mse = torch.tensor(0.0).to(mask_rpn.device)
            

                a = 1  # 损失比例
                rpn_loss = foreground_mse + a*background_mse
                rpn_loss = rpn_loss
                # rpn_loss = self._mask_rpn_loss(mask_label, mask_rpn, T=4, use_weight=False)
                # if count == 1:
                #     rpn_loss = rpn_loss /16
                # if count == 2:
                #     rpn_loss = rpn_loss /4
                # if count == 3:
                #     rpn_loss = rpn_loss

                rpn_losses += rpn_loss
                # print(rpn_loss)
                # -------------------------------------------------- #
                #   利用真实框和先验框获得建议框网络应该有的预测结果
                #   给每个先验框都打上标签
                #   gt_rpn_loc      [num_anchors, 4]
                #   gt_rpn_label    [num_anchors, ]
                # -------------------------------------------------- #
                # gt_rpn_loc, gt_rpn_label    = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
                # gt_rpn_loc                  = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
                # gt_rpn_label                = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()
                # -------------------------------------------------- #
                #   分别计算建议框网络的回归损失和分类损失
                # -------------------------------------------------- #
                # print(rpn_loc.shape)
                # print(gt_rpn_loc.shape)  # 38*38*9, 4
                # print(gt_rpn_label)
                # rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
                # rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
    
                # rpn_loc_loss_all += rpn_loc_loss
                # rpn_cls_loss_all += rpn_cls_loss
                # ------------------------------------------------------ #
                #   利用真实框和建议框获得classifier网络应该有的预测结果
                #   获得三个变量，分别是sample_roi, gt_roi_loc, gt_roi_label
                #   sample_roi      [n_sample, ]
                #   gt_roi_loc      [n_sample, 4]
                #   gt_roi_label    [n_sample, ]
                # ------------------------------------------------------ #
                
                sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
                # print((torch.Tensor(sample_roi).type_as(mask_rpns)).shape)
                # print((torch.ones(len(sample_roi)).type_as(mask_rpns) * roi_indice[0]).shape)
                # print((torch.Tensor(gt_roi_loc).type_as(mask_rpns)).shape)
                # print((torch.Tensor(gt_roi_label).type_as(mask_rpns).long()).shape)
                accumulated_sample_roi.append(torch.Tensor(sample_roi).type_as(mask_rpns))
                accumulated_ones.append(torch.ones(len(sample_roi)).type_as(mask_rpns) * roi_indice[0])
                accumulated_gt_roi_loc.append(torch.Tensor(gt_roi_loc).type_as(mask_rpns))
                accumulated_gt_roi_label.append(torch.Tensor(gt_roi_label).type_as(mask_rpns).long())
                # print(sample_roi.shape)
                # print(gt_roi_loc.shape)
                # print(gt_roi_loc)
            final_sample_roi = torch.cat(accumulated_sample_roi)
            final_ones = torch.cat(accumulated_ones)
            final_gt_roi_loc = torch.cat(accumulated_gt_roi_loc)
            final_gt_roi_label = torch.cat(accumulated_gt_roi_label)
            sample_rois.append(final_sample_roi)
            sample_indexes.append(final_ones)
            gt_roi_locs.append(final_gt_roi_loc)
            gt_roi_labels.append(final_gt_roi_label)
            
        sample_rois     = torch.cat(sample_rois, dim=0)

        sample_indexes  = torch.cat(sample_indexes, dim=0)
        roi_cls_locs, roi_scores = self.model_train([base_feature3, sample_rois, sample_indexes, img_size], mode = 'head')
        # print(roi_cls_locs[0].shape)
        for i in range(n):
            # ------------------------------------------------------ #
            #   根据建议框的种类，取出对应的回归预测结果
            # ------------------------------------------------------ #
            n_sample = roi_cls_locs[i].size()[0]
            
            roi_cls_loc     = roi_cls_locs[i]
            roi_score       = roi_scores[i]
            gt_roi_loc      = gt_roi_locs[i]
            gt_roi_label    = gt_roi_labels[i]
            
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            # print(n_sample)
            # print(gt_roi_label.shape)
            roi_loc     = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # -------------------------------------------------- #
            #   分别计算Classifier网络的回归损失和分类损失
            # -------------------------------------------------- #
            roi_cls_loss, roi_loc_loss = ohem_loss(128, roi_score, gt_roi_label, roi_loc, gt_roi_loc)
            # roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            # roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_loc_loss_all = roi_loc_loss_all + roi_loc_loss
            roi_cls_loss_all = roi_cls_loss_all + roi_cls_loss
        # print(rpn_losses)
        losses = [rpn_losses/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        # print(roi_cls_loss_all)
        # print(rpn_losses)
        losses = losses + [sum(losses)]
        return losses

    def train_step(self, imgs, bboxes, labels, scale, batch ,fp16=False, scaler=None):
        self.optimizer.zero_grad()
        if not fp16:
            losses = self.forward(imgs, bboxes, labels, scale, batch)
            losses[-1].backward()  # 0为只训练mask rpn  -1 为整体 
            self.optimizer.step()
            # print(1)
        else:
            from torch.cuda.amp import autocast
            with autocast():
                losses = self.forward(imgs, bboxes, labels, scale)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(losses[-1]).backward()
            scaler.step(self.optimizer)
            scaler.update()
        # print(losses)
        return losses

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
