import hydra
from omegaconf import DictConfig
import logging
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10 #, Imagenette
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms


from models import twins, SimSiam_network
from tqdm import tqdm
from transform_utils import BarlowTransform


logger = logging.getLogger(__name__)


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


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


train_transform_0 = transforms.Compose([transforms.ToTensor()])
train_transform = transforms.Compose([transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(p=0.5), get_color_distortion(s=0.5), transforms.ToTensor()])
train_transform_big = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(p=0.5), get_color_distortion(s=0.8), transforms.ToTensor()])


def pick_transform(size, aggresiveness, trans_method):
    if trans_method == 'SimCifar':
        train_transform = transforms.Compose([transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip(p=0.5), get_color_distortion(s=0.5), transforms.ToTensor()])
    elif trans_method == 'Barlow':
        train_transform = BarlowTransform(size,aggresive=aggresiveness)
        print('using Barlow Twins augmentation scheme')
    return train_transform
#train_transform_big = Transform()

def torchvision_dataset_stacks(dataset, P, **kwargs):
    '''
    not finished, maybe later?
    '''
    class stacks(dataset):
        def __init__(self, P, **kwargs): 
            super(stacks, self).__init__(**kwargs) 
            self.P = P
            self.transform_0 = transforms.Compose([transforms.ToTensor()])

        def __getitem__(self, idx):
            path, target = self._samples[idx]
            img = Image.open(path).convert("RGB")  # .convert('RGB')
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

class CIFAR100stacks(CIFAR100):
    def __init__(self, P, **kwargs): 
        super(CIFAR100stacks, self).__init__(**kwargs) 
        self.P = P
        self.transform_0 = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        for i in range(2,self.P):
            imgs.append(self.transform(img))
        return torch.stack(imgs), target  # stack a positive pair

# class imagenette320stacks(Imagenette):
#     def __init__(self, P, **kwargs): 
#         super(imagenette320stacks, self).__init__(**kwargs) 
#         self.P = P
#         self.transform_0 = transforms.Compose([transforms.ToTensor()])

#     def __getitem__(self, idx):
#         path, target = self._samples[idx]
#         img = Image.open(path).convert("RGB")  # .convert('RGB')
#         imgs = [self.transform(img), self.transform(img)]
#         for i in range(2,self.P):
#             imgs.append(self.transform(img))
#         return torch.stack(imgs), target  # stack a positive pair

class STL10stacks(STL10):
    def __init__(self, P, **kwargs): 
        super(STL10stacks, self).__init__(**kwargs) 
        self.P = P
        self.transform_0 = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        for i in range(2,self.P):
            imgs.append(self.transform(img))
        return torch.stack(imgs), target  # stack a positive pair

def get_dataloader(dataset, P, data_dir, transform, args, transform0:bool, train:bool):
        if dataset=='cifar10':
            train_set = CIFAR10stacks(P,root=data_dir, train=True, transform=transform, download=True)
        elif dataset=='cifar100':
            print('CIFAR100 is now in use')
            train_set = CIFAR100stacks(P,root=data_dir, train=True, transform=transform, download=True)
        elif dataset=='imagenette320':
            print('imagenette320 is now in use')
            train_set = imagenette320stacks(P,size='320px', root=data_dir, split='train', transform=transform, download=False)
        elif dataset=='STL10':
            print('pre-traning on STL10')
            train_set = STL10stacks(P,root=data_dir, split='unlabeled', transform=transform, download=True)
        
        data_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True
                              , pin_memory=True)
        return data_loader


class nt_xent(object):

    def __init__(self, t=0.5) -> None:
        self.t = t

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        x_scale = x_scores / self.t   # scale with temperature

        # (2N-1)-way softmax without the score of i-th entry itself.
        # Set the diagonals to be large negative values, which become zeros after softmax.
        x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

        # targets 2N elements.
        targets = torch.arange(x.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


class Lbt(object):

    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def __init__(self, batch_size, lambd=0.5) -> None:
        self.lambd = lambd
        self.batch_size = batch_size
        self.off_diagonal = Lbt.off_diagonal

    def forward(self, x):
        shape = x.shape
        x_hat = F.normalize(x.reshape((self.batch_size, 2, *shape[1:])),dim=2)
        z1 = x_hat[:,0,...]
        z2 = x_hat[:,1,...]

        # empirical cross-correlation matrix
        c = z1.T @ z2
        c.div_(self.batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class SimSiamLoss(object):
    'Loss Function of SimSiam'
    def Lss(p, z): # negative cosine similarity
        return -(p*z).sum(dim=1).mean()/2
    
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.criterion = torch.nn.CosineSimilarity(dim=1).cuda()
    
    def forward(self, reps):
        x, y = reps[0], reps[1]
        shape = x.shape
        x_hat = F.normalize(x.reshape((self.batch_size, 2, *shape[1:])),dim=2)
        z1 = x_hat[:,0,...]
        z2 = x_hat[:,1,...]
        y_hat = F.normalize(y.reshape((self.batch_size, 2, *shape[1:])),dim=2)
        p1 = y_hat[:,0,...]
        p2 = y_hat[:,1,...]
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss


class PiCLoss(object):

    def __init__(self, num_views, alpha=1, beta=1, verbose=False) -> None:
        self.num_views = num_views
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

    def forward(self, A):
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

class PiCLoss2(object):

    def __init__(self, num_views, alpha=1, beta=1, verbose=False) -> None:
        self.num_views = num_views
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

    def forward(self, A):
        shape = A.shape ; batchsize = int(shape[0]/self.num_views)
        # A2 shape: batchsize, num_parallel_batch, output_dim
        A_hat = F.normalize(A.reshape((batchsize, self.num_views, *shape[1:])),dim=2)
        var_self = 1 - torch.einsum('ijk,ilk->ijl',A_hat, A_hat)
        loss_self = self.alpha * var_self.mean()
        PV = A_hat.mean(dim=1)
        var_cross = (PV@PV.T - torch.eye(batchsize,device=A.device)).abs().exp()
        loss_cross = self.beta*var_cross.mean()
        loss = loss_self + loss_cross
        if self.verbose :
            print('losses:', round(loss_self.item(),5), round(loss_cross.item(),5))
        return loss

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def train(model, data_loader, optimizer, scheduler, criterion, actual_epoch, args, device='cuda'):

    model.train()

    loss_meter = AverageMeter("SimCLR_loss")
    train_bar = tqdm(data_loader)
    for x, y in train_bar:
        sizes = x.size()
        x = x.view(sizes[0] * sizes[1], sizes[2], sizes[3], sizes[4]).cuda(device, non_blocking=True)
        optimizer.zero_grad()
        feature, rep = model(x)
        loss = criterion(rep)
        
        loss.backward()
        optimizer.step()
        if args.lr_schedule:
            scheduler.step()

        loss_meter.update(loss.item(), x.size(0))
        train_bar.set_description("Train epoch {}, {} loss: {:.4f}".format(actual_epoch, args.method, loss_meter.avg))

    # save checkpoint every log_interval epochs
    if actual_epoch >= args.log_interval and actual_epoch % args.log_interval == 0:
        logger.info("==> Save checkpoint. Train epoch {}, {} loss: {:.4f}".format(actual_epoch, args.method ,loss_meter.avg))
        torch.save(model.state_dict(), 'simclr_{}_epoch{}.pt'.format(args.backbone, actual_epoch))


@hydra.main(version_base='1.1', config_path='conf',config_name='siamese_config.yaml')
def main(args: DictConfig) -> None:
    if args.train_on_gpu1:
        device = 'cuda:1'
        assert args.parallel == False
    else:
        device = 'cuda'
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    ngpus = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {ngpus}")

    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir
    P = 2 if args.method in ['SimCLR', 'Barlow', 'SimSiam'] else args.num_views
  
    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34', 'resnet50']
    base_encoder = eval(args.backbone)
    if args.dataset in ['imagenette160', 'imagenette320']:
        size = 224
    elif args.dataset in ['STL10']:
        size = 96
    elif args.dataset in ['cifar10', 'cifar100']:
        size = 32
    train_transform = pick_transform(size, args.aggresiveness, args.transform)
    data_loader = get_dataloader(args.dataset, P, data_dir, train_transform, args, transform0=False, train=True)

    if args.method == 'SimSiam':
        model = SimSiam_network(base_encoder, size=size)
    else:
        model = twins(base_encoder, size=size, projection_dim=args.projection_dim)
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))
    if args.parallel == True:
        if ngpus > 1:
            model = torch.nn.DataParallel(model)
    model.cuda(device)

    if args.learning_rate_scaling:
        lr = args.learning_rate*args.batch_size/64
    else:
        lr =args.learning_rate
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(data_loader),
            lr,  # lr_lambda computes multiplicative factor
            1e-3))

    if args.method =='SimCLR':
        criterion = nt_xent(t=args.temperature)
    elif args.method == 'Barlow':
        criterion = Lbt(batch_size=args.batch_size, lambd=args.Lbt_Lambda)
    elif args.method == 'SimSiam':
        criterion = SimSiamLoss(batch_size=args.batch_size)
    elif args.method =='PiCCL':
        criterion = PiCLoss(num_views=P, alpha=args.PiCCL_alpha, beta=args.PiCCL_beta)
    elif args.method =='PiCCL2':
        criterion = PiCLoss2(num_views=P, alpha=args.PiCCL_alpha, beta=args.PiCCL_beta)
    criterion = criterion.forward

    for epoch in range(1, args.epochs + 1):
        train(model, data_loader, optimizer, scheduler, criterion, epoch, args, device=device)


if __name__ == '__main__':
    main()





