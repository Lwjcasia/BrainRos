import torch


# weights_best = 'weights/best_voc_origin.pt'
# weights_last = 'weights/last_voc_origin.pt'

# ckpt_b = torch.load(weights_best)
# ckpt_l = torch.load(weights_last)

# print('over')
# from apex import amp
# try:  # Mixed precision training https://github.com/NVIDIA/apex
#     a = 1
# except:
#     print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
#     mixed_precision = False  # not installed
# import os
# batch_size = 16
# print(min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]))
# import torch
# from torch import nn 
# x = torch.tensor([[5,3,2],[1,2,4]])
# timesteps = 100
# precision = 1/timesteps
# c = x.shape[1]
# max_channel_value = x.transpose(0,1).reshape(c,-1).max(dim=1)[0].detach()
# x = x.transpose(1,-1)
# x = x.div(max_channel_value)
# x =  (x/precision).int()*precision
# x  = x.mul(max_channel_value)
# x = x.transpose(1,-1)
# print(x)
# import os
# print(os.getcwd())
import torch

device = torch.device('cuda:{}'.format(6))
print(device)
print(torch.cuda.device_count())