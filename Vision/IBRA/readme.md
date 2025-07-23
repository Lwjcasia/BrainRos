# Integer Binary-Range Alignment Neuron for Spiking Neural Networks


## Introduction

Spiking Neural Networks (SNNs) are noted for their brain-like computation and energy efficiency, but their performance lags behind Artificial Neural Networks (ANNs) in tasks like image classification and object detection due to the limited representational capacity. To address this, we propose a novel spiking neuron, Integer Binary-Range Alignment Leaky Integrate-and-Fire to exponentially expand the information expression capacity of spiking neurons with only a slight energy increase. This is achieved through Integer Binary Leaky Integrate-and-Fire and range alignment strategy. The Integer Binary Leaky Integrate-and-Fire allows integer value activation during training and maintains spike-driven dynamics with binary conversion expands virtual timesteps during inference. The range alignment strategy is designed to solve the spike activation limitation problem where neurons fail to activate high integer values. Experiments show our method outperforms previous SNNs, achieving 74.19% accuracy on ImageNet and 66.2% mAP@50 and 49.1% mAP@50:95 on COCO, surpassing previous bests with the same architecture by +3.45% and +1.6% and +1.8%, respectively. Notably, our SNNs match or exceed ANNs' performance with the same architecture, and the energy efficiency is improved by 6.3 times.

## Dataset

ImageNet


## Get Started


```

cd ImageNet

python -m torch.distributed.launch --nproc_per_node 2 train.py

```

The default time step T=1,D=5.11,N=100, if you want to change the time step and other parameters, please go to the corresponding place in the model.py or train.py file to change it.

## trained models

这里是D=5.11，N=100的训练好的spiking resnet34模型 在 “image\_classification\ImageNet\D=5.11\_N=100\_resnet34.pth”



