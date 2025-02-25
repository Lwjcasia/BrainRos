import torch


weights_best = 'weights/best_voc_origin.pt'
weights_last = 'weights/last_voc_origin.pt'

ckpt_b = torch.load(weights_best)
ckpt_l = torch.load(weights_last)

print('over')