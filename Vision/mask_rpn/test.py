# import torch

# # 创建一个初始张量
# tensor = torch.Tensor([
#     [
#         [[0.1, 0.2], [0.3, 0.4]],  # 第1个通道
#         [[0.5, 0.6], [0.7, 0.8]],  # 第2个通道
#         [[0.9, 1.0], [1.1, 1.2]],  # 第3个通道
#         [[1.3, 1.4], [1.5, 1.6]],  # 第4个通道
#         [[1.7, 1.8], [1.9, 2.0]],  # 第5个通道
#         [[2.1, 2.2], [2.3, 2.4]],  # 第6个通道
#         [[2.5, 2.6], [2.7, 2.8]],  # 第7个通道
#         [[2.9, 3.0], [3.1, 3.2]]   # 第8个通道
#     ]
# ])

# # 执行 permute 操作
# permuted_tensor = tensor.permute(0, 2, 3, 1)

# # 打印原始张量和重新排列后的张量
# print("Original Tensor:")
# print(tensor)
# print("\nPermuted Tensor:")
# print(permuted_tensor)

# # 可选：使用 view 来展示如何改变形状，理解 permute 后的数据结构
# reshaped_tensor = permuted_tensor.contiguous().view(1, -1, 4)
# print("\nReshaped Tensor:")
# print(reshaped_tensor)
# import torch
# import numpy as np
# import cv2

# # 自定义的查找连接区域并返回边界框的函数
# def find_connected_components(x, T):
#     batch_size, num_classes, height, width = x.size()
#     boxes = []

#     for b in range(batch_size):
#         for c in range(num_classes):
#             mask = (x[b, c].cpu().numpy() == T).astype(np.uint8)
#             num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
#             for label in range(1, num_labels):
#                 x_min, y_min, w, h, area = stats[label]
#                 x_max = x_min + w - 1
#                 y_max = y_min + h - 1
#                 boxes.append((b, c, y_min, x_min, y_max, x_max))
    
#     return boxes

# # 示例输入
# T = 1
# input_tensor = torch.tensor([[
#     [[0, 1, 0, 0],
#      [1, 1, 1, 0],
#      [0, 1, 0, 0],
#      [0, 0, 1, 1]],

#     [[0, 0, 0, 0],
#      [0, 1, 0, 0],
#      [0, 1, 1, 1],
#      [0, 0, 0, 0]]
# ]])
# print(input_tensor.shape)
# # 添加 batch 和 num_classes 维度

# # 查找连接区域并返回边界框
# boxes = find_connected_components(input_tensor, T)

# # 打印找到的边界框
# for box in boxes:
#     print(f"Batch: {box[0]}, Class: {box[1]}, Bounding Box: [y_min: {box[2]}, x_min: {box[3]}, y_max: {box[4]}, x_max: {box[5]}]")
# import cv2
# import numpy as np

# # 创建一个2x2的矩阵
# small_matrix = np.array([[0, 2],
#                          [1, 4]], dtype=np.float32)

# # 使用最近邻插值放大矩阵
# upsampled_nearest = cv2.resize(small_matrix, (4, 4), interpolation=cv2.INTER_NEAREST)

# # 使用双线性插值放大矩阵
# upsampled_linear = cv2.resize(small_matrix, (4, 4), interpolation=cv2.INTER_LINEAR)

# # 打印结果
# print("Upsampled with Nearest Neighbor Interpolation:")
# print(upsampled_nearest)
# print("\nUpsampled with Bilinear Interpolation:")
# print(upsampled_linear)
# import torch

# # 创建一个简单的mask_rpn和mask
# mask_rpn = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ], dtype=torch.float32)

# mask = torch.tensor([
#     [0, 4, 0],
#     [4, 0, 0],
#     [0, 0, 4]
# ], dtype=torch.float32)

# # T表示前景的标记值
# T = 4

# # 创建前景掩码
# foreground_mask = (mask == T)

# # 使用前景掩码索引mask_rpn
# selected_values = mask_rpn[foreground_mask]

# print("mask_rpn:", mask_rpn)
# print("mask:", mask)
# print("foreground_mask:", foreground_mask)
# print("Selected values from mask_rpn:", selected_values)
# print(mask[[foreground_mask]])
# foreground_mae = torch.abs(mask_rpn[foreground_mask] - mask[foreground_mask]).mean()
# print(foreground_mae)

# import torch

# # 假设定义
# mask_rpn = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0], [0.0, 8.0, 9.0]])
# mask = torch.tensor([[1.0, 0, 0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# T = 1.0  # 示例中使用1.0作为前景标志

# # 创建前景和后景掩码
# foreground_mask = (mask == T)
# background_mask = (mask == 0)

# # 计算前景的MSE
# foreground_mse = ((mask_rpn[foreground_mask] - mask[foreground_mask]) ** 2).mean()

# # 计算后景的MSE
# background_mse = ((mask_rpn[background_mask] - mask[background_mask]) ** 2).mean()

# print("Foreground MSE:", foreground_mse.item())
# print("Background MSE:", background_mse.item())
# print(93/6)

# import torch

# # 假设 roi_cls_locs 和 roi_indices 已经定义
# # n 表示不同的图像数量（假设是最大索引加1）
# n = 4
# roi_indices = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3])  # 示例数据
# roi_cls_locs = torch.rand(len(roi_indices), 10)  # 假设每个ROI有10个类别位置偏移量

# # 计算每个图像索引的ROI数量
# counts = torch.bincount(roi_indices, minlength=n)

# # 根据每个图像的ROI数量切分roi_cls_locs
# splits = torch.split(roi_cls_locs, counts.tolist())

# # 调整每个切分后的张量的形状，并存储到列表中
# result = []
# for split in splits:
#     reshaped = split.view(-1, roi_cls_locs.size(1))  # 假设你需要保持第二维的大小
#     result.append(reshaped)

# # 输出每个张量的形状以验证
# for idx, tensor in enumerate(result):
#     print(f"Tensor {idx} shape: {tensor.shape}")

# import torch
# from nets.frcnn_fpn import FasterRCNN
# import numpy as np

# from utils.utils import (get_classes, seed_everything, show_config,
#                          worker_init_fn)

# model_path      = 'pre_model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
# classes_path    = 'model_data/voc_classes.txt'
# class_names, num_classes = get_classes(classes_path)
# anchors_size    = [8, 16, 32]
# backbone        = "resnet50"
# pretrained      = False
# model = FasterRCNN(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)
# if model_path != '':
#     #------------------------------------------------------#
#     #   权值文件请看README，百度网盘下载
#     #------------------------------------------------------#
#     print('Load weights {}.'.format(model_path))
    
#     #------------------------------------------------------#
#     #   根据预训练权重的Key和模型的Key进行加载
#     #------------------------------------------------------#

#     device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model_dict      = model.state_dict()

#     pretrained_dict = torch.load(model_path, map_location = device)
#     # print(pretrained_dict)
#     # load_key, no_load_key, temp_dict = [], [], {}
#     # for k, v in pretrained_dict.items():
#     #     if k in name_map:  # 确保映射存在
#     #         k = name_map[k]
#     #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#     #         temp_dict[k] = v
#     #         load_key.append(k)
#     #     else:
#     #         no_load_key.append(k)
#     load_key, no_load_key, temp_dict = [], [], {}
#     for k, v in pretrained_dict.items():
#         print(k)
#         if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#             temp_dict[k] = v
#             load_key.append(k)
#         else:
#             no_load_key.append(k)
#     model_dict.update(temp_dict)
#     model.load_state_dict(model_dict)
#     #------------------------------------------------------#
#     #   显示没有匹配上的Key
#     #------------------------------------------------------#
#     print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
#     print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
#     print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")



# import torch

# # 创建一个tensor
# sample_roi = [1, 2, 3]
# sample_roi2 = [1, 2, 3]
# mask_rpns = torch.randn(5, 5)

# # 转换为和mask_rpns相同的类型
# tensor = torch.Tensor(sample_roi).type_as(mask_rpns)
# tensor2 = torch.Tensor(sample_roi2).type_as(mask_rpns)
# # 放入列表
# accumulated_sample_roi = [tensor]
# accumulated_sample_roi.append(tensor2)
# # 使用torch.cat
# final_sample_roi = torch.cat(accumulated_sample_roi)

# print(final_sample_roi)


import torch

# 创建张量 A 和 B
A = torch.randn(2, 4, 2)
B = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)

# 执行广播并进行元素级乘法
C = A * B

print("Shape of A:", A)
print("Shape of B after broadcasting:", B)
print("Shape of C:", C)