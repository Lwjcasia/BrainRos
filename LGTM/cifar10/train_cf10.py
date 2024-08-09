import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode 
from calflops import calculate_flops
import pdb 

class ClassificationPresetTrain:
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

from torch import Tensor
from typing import Tuple
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

def to_c_last(x_seq: torch.Tensor):
    if x_seq.dim() == 3:
        # [T, N, C]
        return x_seq
    else:
        return x_seq.transpose(2, -1)

class IFNode5(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)

def create_neuron(neu: str, **kwargs):
    if neu == 'if5':
        return IFNode5(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'])

def shift_channels(x):
    # print("*******shift_channels : {}".format(x.size()))
    t, b, c, _, _ = x.size() 
    out = torch.zeros_like(x)
    out[:-1, :, :c//2] = x[1:, :, :c//2]  # shift left
    out[:, :, c//2:] = x[:, :, c//2:]  # not shift
    return out 

class LModule(nn.Module):
    def __init__(self, in_channels, mid_channels) -> None:
        super(LModule, self).__init__()
        self.in_channels = in_channels 
        self.mid_channels = mid_channels 

        self.conv_short1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        self.bn_short1 = nn.BatchNorm2d(mid_channels)
        self.conv_short2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        self.bn_short2 = nn.BatchNorm2d(mid_channels)
        self.conv_short3 = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1)
        self.bn_short3 = nn.BatchNorm2d(in_channels)

        self.active = nn.Sigmoid() 
        
    def forward(self, x):
        features = shift_channels(x)
        features = functional.seq_to_ann_forward(features, [self.conv_short1, self.bn_short1]) 
        features = functional.seq_to_ann_forward(features, [self.conv_short2, self.bn_short2])
        features = functional.seq_to_ann_forward(features, [self.conv_short3, self.bn_short3])

        features = self.active(features) 
        out = features * x 
        return out 
        
class GModule(nn.Module):
    def __init__(self, time, mode='cut') -> None:
        super(GModule, self).__init__()
        self.mode = mode 

        self.conv_long = nn.Conv2d(in_channels=time, out_channels=time, kernel_size=3, stride=1, padding=1)
        self.bn_long = nn.BatchNorm2d(time)
        
        self.active = nn.Sigmoid() 

    def forward(self, x):
        # pdb.set_trace() 
        features = x 
        t, b, c, h, w = features.size() 

        if self.mode == 'cut': 
            # features, _ = torch.max(features, dim=2) # t, b, h, w
            features = torch.mean(features, dim=2) # t, b, h, w
            features = features.transpose(0, 1).contiguous() # b, t, h, w
            features = self.conv_long(features)
            features = self.bn_long(features)
            features = features.transpose(0, 1).contiguous().unsqueeze(2).repeat(1,1,c,1,1) # t, b, c, h, w 
        elif self.mode == 'per': 
            features = features.permute(1, 2, 0, 3, 4).contiguous().view(-1, t, h, w) # t, b, c, h, w -> b*c, t, h, w
            features = self.conv_long(features)
            features = self.bn_long(features)
            features = features.view(b, c, -1, h, w).permute(2, 0, 1, 3, 4).contiguous() # b*c, t, h, w -> b, c, t, h, w -> t, b, c, h, w 
        else:
            raise NotImplementedError(self.mode) 
        
        features = self.active(features) 
        out = features * x 
        return out 

class LGModule(nn.Module):
    def __init__(self, in_channels, mid_channels, time, mode='trans') -> None:
        super(LGModule, self).__init__()

        self.LM = LModule(in_channels=in_channels, mid_channels=mid_channels)
        self.GM = GModule(time=time, mode=mode)
       
    def forward(self, x):
        out = x
        # out_l = self.LM(out)
        # out_g = self.GM(out)
        # out = out_l + out_g

        out = self.LM(out)
        out = self.GM(out)
        return out 
    
class CIFAR10Net(nn.Module):
    def __init__(self, channels, neu: str, T: int):
        super(CIFAR10Net, self).__init__()
        conv = []

        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                # conv.append(LGModule(in_channels=channels, mid_channels=channels//8, time=T, mode='cut'))
                conv.append(create_neuron(neu, surrogate_function=surrogate.ATan(), T=T))
                # conv.append(LGModule(in_channels=channels, mid_channels=channels//32, time=T, mode='cut'))
            # conv.append(LGModule(in_channels=channels, mid_channels=channels//16, time=T, mode='cut'))
            conv.append(layer.AvgPool2d(2, 2))
            conv.append(LGModule(in_channels=channels, mid_channels=channels//8, time=T, mode='cut'))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * 8 * 8, channels * 8 * 8 // 4),
            create_neuron(neu, features=channels * 8 * 8 // 4, surrogate_function=surrogate.ATan(), T=T),
            layer.Linear(channels * 8 * 8 // 4, 10),
        )
        functional.set_step_mode(self, 'm')

    def forward(self, x_seq: torch.Tensor):
        # pdb.set_trace()
        # x_seq.shape : [T, N, C, H, W]
        # print("forward x_seq.shape : {}".format(x_seq.shape))

        # out = self.conv(x_seq)
        # out = self.fc(out)
        # return out 
        # return self.fc(self.conv(x_seq))

        # test firing num loss 
        firing_num = 0
        out = x_seq
        # pdb.set_trace()
        for conv_layer in self.conv:
            out = conv_layer(out)
            if isinstance(conv_layer, IFNode5):
                # pdb.set_trace()
                firing_num += out.sum().item()

        for fc_layer in self.fc:
            if isinstance(fc_layer, IFNode5):
                out = fc_layer(out)
                firing_num += out.sum().item()
            else:
                out = fc_layer(out)
        return out, firing_num

def evaluate(net, criterion, test_data_loader, device, T=4):
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                if T > 1:
                    # pdb.set_trace()
                    img = img.unsqueeze(0).repeat(T, 1, 1, 1, 1)
                    y, firing_num = net(img)
                    y = y.mean(0)
                else:
                    y = net(img)
                loss = F.cross_entropy(y, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (y.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        # test_time = time.time()
        # test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        print(f"Test Loss: {test_loss:.4f}, Test Acc Top1: {test_acc:.4f}")
        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)


def main():
    parser = argparse.ArgumentParser(description='Classify CIFAR10')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size') 
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', 
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of the CIFAR10 dataset')
    parser.add_argument('-out-dir', type=str, default='./logs_cf10', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or AdamW')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-T', default=4, type=int, help='number of time-steps')
    parser.add_argument('-test_only', dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("-loss_lambda", type=float, default=0) # 1e-6 

    args = parser.parse_args()
    print(args)

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(10, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(10, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    transform_train = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                  std=(0.2023, 0.1994, 0.2010), interpolation=InterpolationMode('bilinear'),
                                                  auto_augment_policy='ta_wide',
                                                  random_erase_prob=0.1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=transform_train,
            download=True)

    test_set = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    out_dir = f'T{args.T}_e{args.epochs}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}'

    if args.amp:
        out_dir += '_amp_test' # lg6_lsbr32*_gmean*_afterneu

    pt_dir = os.path.join(args.out_dir, 'pt', out_dir)
    out_dir = os.path.join(args.out_dir, out_dir)
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    net = CIFAR10Net(args.channels, 'if5', args.T)

    # output model structure and calculate params, flops
    # print(net)
    # input_shape = (args.T, args.b, 3, 32, 32)
    # flops, macs, paras = calculate_flops(net, input_shape, output_as_string=True, output_precision=4)
    # print(f"calculate_flops: flops: {flops}, macs: {macs}, params: {paras}")

    # pdb.set_trace()

    net.to(args.device)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd': 
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print(f'Resume from {args.resume}, epoch = {start_epoch}, max_test_acc = {max_test_acc}.')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.') 

    # for test only
    if args.test_only:
        criterion = F.cross_entropy
        evaluate(net, criterion, test_data_loader, device=args.device)
        return

    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        firing_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                if args.T > 1:
                    img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                    # y = net(img).mean(0)
                    y, firing_num = net(img)
                    y = y.mean(0)
                else:
                    y = net(img)

                # loss = F.cross_entropy(y, label, label_smoothing=0.1)
                firing = args.loss_lambda * firing_num
                loss = F.cross_entropy(y, label, label_smoothing=0.1) + firing

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            firing_loss += firing * label.shape[0]

            train_acc += (y.argmax(1) == label.argmax(1)).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        firing_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                if args.T > 1:
                    # pdb.set_trace()
                    img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
                    # y = net(img).mean(0)
                    y, firing_num = net(img)
                    y = y.mean(0)
                else:
                    y = net(img)
                loss = F.cross_entropy(y, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (y.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, firing_loss ={firing_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')



if __name__ == '__main__':
    main()