import hydra
from omegaconf import DictConfig
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, STL10#, Imagenette
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from models import twins, SimSiam_network
from tqdm import tqdm


logger = logging.getLogger(__name__)

HYDRA_FULL_ERROR=1
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


def run_epoch(model, dataloader, epoch, device='cuda',optimizer=None, scheduler=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
        x, y = x.cuda(device), y.cuda(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(version_base='1.1', config_path='conf',config_name='siamese_config.yaml')
def finetune(args: DictConfig) -> None:
    print('load_epoch:',args.load_epoch)
    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    train_transform_big = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    data_dir = hydra.utils.to_absolute_path(args.data_dir)

    device='cuda:2' if args.train_on_gpu1 else 'cuda'
    if args.dataset=='cifar10':
        train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
        test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)
        in_momory=False
        size = 32
        print('testing on CIFAR-10')
    elif args.dataset=='cifar100':
        train_set = CIFAR100(root=data_dir, train=True, transform=train_transform, download=False)
        test_set = CIFAR100(root=data_dir, train=False, transform=test_transform, download=False)
        in_momory=False
        size = 32
        print('testing on CIFAR-100')
    elif args.dataset=='imagenette320':
        print('imagenette320 is now in use')
        # train_set = Imagenette(size='320px', root=data_dir, split='train', transform=train_transform_big, download=False)
        # test_set = Imagenette(size='320px', root=data_dir, split='val', transform=train_transform_big, download=False)
        in_momory=False
        size = 224
    elif args.dataset == 'STL10':
        print('FineTuning and Testing on STL-10')
        train_set = train_set = STL10(root=data_dir, split='train', transform=train_transform_big, download=True)
        test_set = train_set = STL10(root=data_dir, split='test', transform=train_transform_big, download=True)
        in_momory=False
        size = 96
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=in_momory) #, sampler=sampler
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=in_momory)

    # Prepare model
    base_encoder = eval(args.backbone)
    if args.method == 'SimSiam':
        pre_model = SimSiam_network(base_encoder, size=size, dim=2048, pred_dim=512).cuda(device)
    else:
        pre_model = twins(base_encoder,size=size, projection_dim=args.projection_dim).cuda(device)
    print(pre_model.enc.conv1.weight.shape)
    checkpoint_dir = '/home/jmyan/Project/PiCCL/logs/PiCCL/cifar100/simclr_resnet18_epoch50.pt'
    # checkpoint_dir = '/home/yimingk/PiCCL-CIFAR10-clean/logs/{}/{}/simclr_{}_epoch{}.pt'.format(args.method, args.dataset, args.backbone, args.load_epoch)
    checkpoint = torch.load(checkpoint_dir)
    print('loading pretrained model from: ' + checkpoint_dir)
    if any('module' in key for key in checkpoint.keys()):
        print('Model trained using DP or DDP')
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    #if args.parallel==True:
    #    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}


    pre_model.load_state_dict(checkpoint)
    feature_dim = 512 # if args.method == 'SimSiam' else args.projection_dim
    num_classes = 100 if args.dataset == 'cifar100' else 10
    model = LinModel(pre_model.enc, feature_dim=feature_dim, n_classes=num_classes)
    model = model.cuda(device)

    # Fix encoder
    model.enc.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.
    # optimizer = Adam(parameters, lr=0.001)

    lr = 0.02 if args.dataset=='imagenette320' else 0.2
    optimizer = torch.optim.SGD(
        parameters,
        lr,   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
        momentum=args.momentum,
        weight_decay=0.,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-3))

    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, device, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch, device)

        if test_acc > optimal_acc:
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            torch.save(model.state_dict(), 'simclr_lin_{}_best.pth'.format(args.backbone))
        '''
        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            torch.save(model.state_dict(), 'simclr_lin_{}_best.pth'.format(args.backbone))
        '''

    logger.info("Best Test Acc: {:.4f}".format(optimal_acc))


if __name__ == '__main__':
    finetune()



