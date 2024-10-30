import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


# def create_foreground_weight_map(mask_label, T, max_weight=10, min_weight=1):
#     # 计算前景区域
#     foreground = (mask_label == T).astype(np.uint8)  # 前景为1
#     # 如果全是前景，增加一圈0（背景）
#     if np.all(foreground == 1):
#         padded_foreground = np.pad(foreground, pad_width=1, mode='constant', constant_values=0)
#     else:
#         padded_foreground = foreground
#     # 距离变换：计算每个前景点到最近背景点的距离
#     distances = scipy.ndimage.distance_transform_edt(padded_foreground == 1) - 1 # 前景到背景点最近的距离，也就是到边界框最近的距离

#     if np.all(foreground == 1):
#         distances = distances[1:-1, 1:-1]
#     # 归一化距离：转换为权重，使权重在min_weight和max_weight之间线性变化
#     max_dist = np.max(distances)
#     if max_dist > 0:
#         # 归一化距离，并线性映射到权重范围
#         weight_map = max_weight - (max_weight - min_weight) * distances / max_dist
#     else:
#         # 所有前景点都在边界上（不太可能，但以防万一）
#         weight_map = np.full_like(distances, max_weight)

#     # 为背景设置最小权重
#     weight_map[foreground == 0] = 0

#     return weight_map


# def create_background_weight_map(mask_label, T, max_weight=10, min_weight=1):
#     # 计算背景区域
#     background = (mask_label == 0).astype(np.uint8)  # 背景为1

#     # 距离变换：计算每个背景点到最近前景点的距离
#     distances = scipy.ndimage.distance_transform_edt(background == 1) # True 到 False的最近距离

#     # 归一化距离：转换为权重，使权重在min_weight和max_weight之间线性变化
#     max_dist = np.max(distances)
#     if np.all(background == 1):
#         # 全为背景，设定权重为最小权重
#         weight_map = np.full_like(distances, 0)
#     else:
#         if max_dist > 0:
#             # 归一化距离，并线性映射到权重范围
#             weight_map = min_weight + (max_weight - min_weight) * distances / max_dist
#         else:
#             weight_map = np.full_like(distances, min_weight)

#     weight_map[background == 0] = 0
#     return weight_map



# background = np.zeros((100, 100), dtype=np.uint8)
# background[0:100, 0:100] = 1  # 定义一个前景区域
# # background[58:80, 15:35] = 1
# # background[34:78, 44:86] = 1

# foreground_weight_map = create_foreground_weight_map(background, 1)
# background_weight_map = create_background_weight_map(background, 1)

# # 结合前景和后景权重
# combined_weight_map = foreground_weight_map + background_weight_map

# # print(combined_weight_map)
# # 可视化
# plt.figure(figsize=(18, 6))
# plt.subplot(1, 4, 1)
# plt.title("Foreground")
# plt.imshow(background, cmap='gray', interpolation='nearest')
# plt.colorbar()

# plt.subplot(1, 4, 2)
# plt.title("Foreground Weight Map")
# plt.imshow(foreground_weight_map, cmap='viridis', interpolation='nearest')
# plt.colorbar()

# plt.subplot(1, 4, 3)
# plt.title("Background Weight Map")
# plt.imshow(background_weight_map, cmap='viridis', interpolation='nearest')
# plt.colorbar()

# plt.subplot(1, 4, 4)
# plt.title("Combined Weight Map")
# plt.imshow(combined_weight_map, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.show()
# import torch
# def create_foreground_weight_map(mask_label, T, max_weight=10, min_weight=0.1):
#     # 确保mask_label是PyTorch Tensor
#     if not isinstance(mask_label, torch.Tensor):
#         mask_label = torch.tensor(mask_label, dtype=torch.uint8)

#     # 计算前景区域
#     foregrounds = (mask_label == T)

#     padded_foregrounds = []
#     distances = []
#     # 遍历每个channel的前景
#     for foreground in foregrounds:
#         foreground_np = foreground.cpu().numpy().astype(np.uint8)
#         # 如果全是前景，增加一圈0（背景）
#         if np.all(foreground_np == 1):
#             padded_foreground = np.pad(foreground_np, pad_width=1, mode='constant', constant_values=0)

#         else:
#             padded_foreground = foreground_np
#         padded_foregrounds.append(padded_foreground)

#         # 计算距离变换
#         distance = scipy.ndimage.distance_transform_edt(padded_foreground == 1) - 1
#         if np.all(foreground_np == 1):
#             distance = distance[1:-1, 1:-1]
#         distances.append(distance)

#     weight_maps = []
#     for idx, distance in enumerate(distances):
#         # 归一化距离：转换为权重，使权重在min_weight和max_weight之间线性变化
#         max_dist = np.max(distance)
#         if max_dist > 0:
#             weight_map = max_weight - (max_weight - min_weight) * distance / max_dist
#         else:
#             weight_map = np.full_like(distance, max_weight)

#         # 转换为Tensor并为背景设置权重0
#         weight_map = torch.from_numpy(weight_map).float()
#         weight_map[foregrounds[idx] == 0] = 0
#         weight_maps.append(weight_map)

#     return torch.stack(weight_maps)




# def create_background_weight_map(mask_label, T, max_weight=10, min_weight=1):
#     # 确保mask_label是PyTorch Tensor
#     if not isinstance(mask_label, torch.Tensor):
#         mask_label = torch.tensor(mask_label, dtype=torch.uint8)
    
#     # 计算背景区域
#     background = (mask_label == 0)

#     weight_maps = []
#     for i in range(background.shape[0]):  # 遍历每个通道
#         foreground_np = background[i].cpu().numpy().astype(np.uint8)

#         # 距离变换：计算每个背景点到最近前景点的距离
#         distances = scipy.ndimage.distance_transform_edt(foreground_np == 1)

#         # 检查是否全是背景
#         if np.all(foreground_np == 1):
#             # 全为背景，设定权重为最小权重
#             weight_map = np.full_like(distances, 0)
#         else:
#             # 归一化距离：转换为权重，使权重在min_weight和max_weight之间线性变化
#             max_dist = np.max(distances)
#             if max_dist > 0:
#                 # 归一化距离，并线性映射到权重范围
#                 weight_map = min_weight + (max_weight - min_weight) * distances / max_dist
#             else:
#                 # 所有背景点都在前景边界上（不太可能，但以防万一）
#                 weight_map = np.full_like(distances, min_weight)

#         # 为前景设置权重为0
#         weight_map[foreground_np == 0] = 0
#         weight_maps.append(weight_map)

#     # 将权重图列表转换为Tensor
#     weight_maps_tensor = torch.from_numpy(np.stack(weight_maps)).float()
#     return weight_maps_tensor


# mask_label = torch.tensor([
#     [
#         [0, 0, 0, 0, 0],  # 完全背景
#         [0, 1, 1, 1, 0],
#         [0, 1, 1, 1, 0],
#         [0, 1, 1, 1, 0],
#         [0, 0, 0, 0, 0]
#     ],
#     [
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]
#     ],
#     [
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1]
#     ]
# ], dtype=torch.uint8)  # 前景为1，背景为0

# weight_maps = create_foreground_weight_map(mask_label, 1, max_weight=10, min_weight=1)

# # 绘图
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for i, ax in enumerate(axes.flat):
#     im = ax.imshow(weight_maps[i], cmap='viridis', interpolation='nearest')
#     ax.set_title(f'Channel {i+1}')
#     fig.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()
