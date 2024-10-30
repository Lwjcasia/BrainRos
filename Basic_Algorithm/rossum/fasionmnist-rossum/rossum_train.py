import torch
import torchvision
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
from all_conv_rossum import SCNN
from torch.nn.modules.loss import _Loss
import matplotlib.pyplot as plt



train_dir='F:\pycharmproject\spiking_unet11\spiking_unet\one_more_try/fenleiceshiok/fenleiceshi/train/'
test_dir='F:\pycharmproject\spiking_unet11\spiking_unet\one_more_try/fenleiceshiok/fenleiceshi/test/'
batch_size=64
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time_window=10
num_class=10
torch.autograd.set_detect_anomaly(True)


# ohe = OneHotEncoder()
# ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14]])
# # print(ohe.transform(np.array(3).reshape(1,1)).toarray())
# label=torch.tensor([0,1,2,3,4,5,6,7,8,9])
# def time_to_first_coding(label):
#     target_seq=[]
#     for i in range(len(label)):
#
#         one_label_i = ohe.transform(label[i].cpu().numpy().reshape(1,1)).toarray()
#         index=np.argwhere(one_label_i == 1)[0][1]
#         first=one_label_i[0][:index+1]
#         last=one_label_i[0][index+1:]
#
#         last[-5:]=1
#         random.shuffle(last)
#
#         target_seq.append(np.hstack((first,last)))
#
#     return target_seq

ohe = OneHotEncoder()
ohe.fit([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
def one_hot_coding(label):
    target_seq=[]
    for i in range(len(label)):

        one_label_i = ohe.transform(label[i].cpu().numpy().reshape(1,1)).toarray()


        target_seq.append(one_label_i)

    return target_seq



def vrdistance(f, g, tau): #f是网络输出 g是label   (T,batch,10)
    # Computes the Van Rossum distance between f and g
    f=f.permute(1,2,0)   #(batch,10,T)
    g = g.permute(1, 2, 0)
    assert f.shape == g.shape, 'Spike trains must have the same shape, had f: ' + str(f.shape) + ', g: ' + str(g.shape)

    kernel = torch.FloatTensor(np.exp([-t / tau for t in range(f.shape[-1])])).flip(0).cuda()  # Flip because conv1d computes cross-correlations (T,1)
    # kernels=kernel.repeat((15,1)).permute(1,0)  #(10,T)


    diff = (f - g).to(torch.float32)
    # q=torch.matmul(diff, kernel)   [batch,10]
    # diff=diff.permute(1,2,3,0)
    return torch.norm(torch.matmul(diff, kernel), dim=-1)


class VRDistance(_Loss):
    # Van Rossum distance loss
    def __init__(self, tau, size_average=None, reduce=None, reduction='mean'):
        super(VRDistance, self).__init__(size_average, reduce, reduction)
        self.tau = tau

    def forward(self, input, target):
        return vrdistance(input, target, self.tau)


def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__=='__main__':

    train_data_dataset = torchvision.datasets.MNIST(
            root=train_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=False)
    train_data_loader = torch.utils.data.DataLoader(
            train_data_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8)

    test_data_dataset= torchvision.datasets.MNIST(
                root=test_dir,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=False)
    test_data_loader = torch.utils.data.DataLoader(
            dataset=test_data_dataset,
            batch_size=100,
            shuffle=True,
            drop_last=False,
            num_workers=8)

    # def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=5):
    #     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    #     if epoch % lr_decay_epoch == 0 and epoch > 1:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = param_group['lr'] * 0.1
    #     return optimizer

    net = SCNN().cuda()
    # net.load_state_dict(torch.load('sfcn_parms.pth'))

    loss_function = nn.CrossEntropyLoss().cuda()
    # 使用Adam优化器
    lr=0.001
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    rossum_distance = VRDistance(tau=1)   #好像tau小点效果好
    best_acc = 0.0
    train_epoch=1
    # coding_seq=torch.tensor(time_to_first_coding(label),device=device)  #存放的是0-9的一次脉冲编码的结果
    plt_tloss=[]
    plt_tacc=[]
    plt_eloss=[]
    plt_eacc=[]
    for epoch in range(train_epoch):
        net.train()
        losses = []
        correct = 0.0
        total = 0.0
        adjust_learning_rate(optimizer,epoch,start_lr=lr)
        for batch, (img, label) in enumerate(train_data_loader):
            img = img.to(device)
            # label_seq=one_hot_coding(label)
            # label_seq=torch.repeat()
            y_one_hot = torch.zeros(batch_size, 10).scatter_(1, label.reshape((label.shape[0],1)), 1)
            label_seq=y_one_hot.repeat(time_window,1,1)

            optimizer.zero_grad()
            out = net(img)  #out shape[time_window,batch,10]
            # print(out.T)

            loss = rossum_distance(out, label_seq.to(device))  #loss的大小为【batch】
            loss=torch.mean(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            out=torch.sum(out,dim=0)   ##[batch,10]
            correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()

            total += out.shape[0]
            if batch % 100 == 0 and batch!=0:
                acc = correct / total
                plt_tacc.append(acc)
                plt_tloss.append(np.array(losses).mean())
                print('Epoch %d [%d/%d] ANN Training Loss:%.3f Accuracy:%.3f' % (epoch,
                                                                                 batch + 1,
                                                                                 len(train_data_loader),
                                                                                 np.array(losses).mean(),
                                                                                 acc))
                correct = 0.0
                total = 0.0

        # optimizer = lr_scheduler(optimizer, epoch, lr, 3)

        net.eval()
        correct = 0.0
        total = 0.0
        losses = []
        with torch.no_grad():
            for batch, (img, label) in enumerate(test_data_loader):
                img = img.to(device)
                y_one_hot = torch.zeros(100, 10).scatter_(1, label.reshape((label.shape[0], 1)), 1)
                label_seq = y_one_hot.repeat(time_window, 1, 1)

                out = net(img)  # out shape[time_window,batch,10]
                # print(out.T)

                loss = rossum_distance(out, label_seq.to(device))  # loss的大小为【batch】
                loss=torch.mean(loss)
                out = torch.sum(out, dim=0)  #[batch,10]
                correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()

                total += out.shape[0]
                losses.append(loss.item())
            acc = correct / total
            if epoch == None:
                print('ANN Validating Accuracy:%.3f' % (acc))
            else:
                plt_eacc.append(acc)
                plt_eloss.append(np.array(losses).mean())
                print('Epoch %d [%d/%d] ANN Validating Loss:%.3f Accuracy:%.3f' % (epoch,
                                                                                   batch + 1,
                                                                                   len(test_data_loader),
                                                                                   np.array(losses).mean(),
                                                                                   acc))

        if best_acc <= acc:
            best_acc=acc
            torch.save(net, 'best_snn_%d.pkl'%acc)

    plt.plot(plt_tloss,'r')
    plt.xlabel('epoch')
    plt.title('train_loss')
    plt.show()


    plt.plot(plt_eloss, 'g')
    plt.xlabel('epoch')
    plt.title('eval_loss')
    plt.show()

    plt.plot(plt_tacc,'r')
    plt.xlabel('epoch')
    plt.title('train_acc')
    plt.show()


    plt.plot(plt_eacc,'g')
    plt.xlabel('epoch')
    plt.title('eval_acc')
    plt.show()