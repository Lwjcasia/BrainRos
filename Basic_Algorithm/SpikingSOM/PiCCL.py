import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10
from PIL import Image


class MNISTstacks(MNIST):
    def __init__(self, P, **kwargs): 
        super(MNISTstacks, self).__init__(**kwargs) 
        self.P = P

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img.numpy(), mode="L")  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        for i in range(2,self.P):
            imgs.append(self.transform(img))
        return torch.stack(imgs), target  # stack a positive pair


class CIFAR10stacks(CIFAR10):
    def __init__(self, P, **kwargs): 
        super(CIFAR10stacks, self).__init__(**kwargs) 
        self.P = P

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        for i in range(2,self.P):
            imgs.append(self.transform(img))
        return torch.stack(imgs), target  # stack a positive pair


class ModelAddProjector(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.enc = encoder
        self.projector = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        shape=x.shape
        x = x.view(shape[0]*shape[1],*shape[2:])
        x = self.enc(x)
        x = self.projector(x)
        return x


class PiCLoss(object):

    def __init__(self, num_views, alpha=1, beta=1, verbose=False) -> None:
        self.num_views = num_views
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

    def forward(self, A, target):
        shape = A.shape ; batchsize = int(shape[0]/self.num_views)
        # A2 shape: batchsize, num_parallel_batch, output_dim
        A_hat = F.normalize(A.reshape((batchsize, self.num_views, *shape[1:])),dim=2)
        var_self = 1 - torch.einsum('ijk,ilk->ijl',A_hat, A_hat)
        loss_self = self.alpha * var_self.mean()
        PV = F.normalize(A_hat.mean(dim=1),dim=1)
        var_cross = (PV@PV.T - torch.eye(batchsize,device=A.device)).abs().exp()
        loss_cross = self.beta*var_cross.mean()
        loss = loss_self + loss_cross
        if self.verbose :
            print('losses:', round(loss_self.item(),5), round(loss_cross.item(),5))
        return loss


class twins(nn.Module):

    def __init__(self, base_encoder, size=32, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(weights=None)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        if size==32:
            shape = (3, 64, 3, 1, 1)
            print(f"picture size is 32 by 32, will change conv1 to {shape[2]}*{shape[2]} stride {shape[3]}, removing maxpooling layer")
            self.enc.conv1 = nn.Conv2d(*shape, bias=False)
            self.enc.maxpool = nn.Identity()
        elif size==96:
            shape = (3, 64, 5, 2, 1)
            print(f"picture size is 96 by 96, will change conv1 to {shape[2]}*{shape[2]} stride {shape[3]}")
            self.enc.conv1 = nn.Conv2d(*shape, bias=False)
            #self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection



