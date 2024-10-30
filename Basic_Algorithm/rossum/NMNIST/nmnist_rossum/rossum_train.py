import logging

import matplotlib.pyplot as plt
import loadNMNIST_Spiking
import torch.nn as nn

from all_conv_rossum import SCNN
from torch.nn.modules.loss import _Loss
from get_dataset import *



batch_size=50
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time_window=25
num_class=10
torch.autograd.set_detect_anomaly(True)


def psp(inputs):
    shape = inputs.shape

    tau_s = 3

    syn = torch.zeros(shape[0], shape[1]).cuda()
    syns = torch.zeros(shape[0], shape[1], time_window).cuda()

    for t in range(time_window):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s

    return syns


def vrdistance(f, g): #f是网络输出 g是label   (T,batch,10)
    # Computes the Van Rossum distance between f and g
    f=f.permute(1,2,0)   #(batch,10,T)
    g = g.permute(1, 2, 0)
    assert f.shape == g.shape, 'Spike trains must have the same shape, had f: ' + str(f.shape) + ', g: ' + str(g.shape)

    diff = (psp(f) - psp(g)).to(torch.float32)

    return (1/2*torch.sum(diff ** 2))/batch_size  #torch.matmul(diff,kernel).shape=[64,10]



class VRDistance(_Loss):
    # Van Rossum distance loss
    def __init__(self,  size_average=None, reduce=None, reduction='mean'):
        super(VRDistance, self).__init__(size_average, reduce, reduction)


    def forward(self, input, target):
        return vrdistance(input, target)

# def adjust_learning_rate(optimizer, epoch, start_lr):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = start_lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

weight_noise=[]
def compute_weight_noise(new_dict,last_dict):
    sum=0
    for param in new_dict:  #param是参数名称
        sum+=torch.sum(torch.abs(new_dict['{}'.format(param)]-last_dict['{}'.format(param)]))

    return sum


if __name__=='__main__':
    data_path='../NMNIST_data/2312_3000_stable/'

    train_loader, test_loader = loadNMNIST_Spiking.get_nmnist(data_path,n_steps=time_window,batch_size=batch_size)


    net = SCNN().cuda()
    torch.save(net.state_dict(),'last_state.pth')

    logging.basicConfig(filename='result.log',level=logging.INFO)

    loss_function = nn.CrossEntropyLoss().cuda()
    loss_function=nn.MSELoss()
    # 使用Adam优化器
    lr=0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    rossum_distance = VRDistance()   #好像tau小点效果好
    best_acc = 0.0
    train_epoch=50

    logging.info('dataset name:{}\n time window:{}\n epoch:{}\n batch size:{}\n lr:{}'.format('NMINST',time_window,train_epoch,batch_size,lr))


    # coding_seq=torch.tensor(time_to_first_coding(label),device=device)  #存放的是0-9的一次脉冲编码的结果
    plt_tacc=[]
    plt_tloss=[]
    plt_eacc=[]
    plt_eloss=[]
    for epoch in range(train_epoch):
        # logging.info('epoch:{}'.format(epoch))
        last_state=torch.load('last_state.pth')

        net.train()
        losses = []
        correct = 0.0
        total = 0.0


        for batch, (img, label) in enumerate(train_loader):
            img = img.to(device)
            # label_seq=one_hot_coding(label)
            # label_seq=torch.repeat()
            y_one_hot = torch.zeros(batch_size, 10).scatter_(1, label.reshape((label.shape[0],1)), 1)
            label_seq=y_one_hot.repeat(time_window,1,1)

            optimizer.zero_grad()
            out = net(img)  #out shape[time_window,batch,10]
            # print(out.T)

            loss = rossum_distance(out, label_seq.to(device))  #loss的大小为【batch】
            #loss=loss_function(out, label_seq.to(device))
            #loss=torch.mean(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            out=torch.sum(out,dim=0)   ##[batch,10]
            correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()

            total += out.shape[0]
            if batch % 100 == 0 and batch!=0:
                acc = correct / total

                print('Epoch %d [%d/%d] ANN Training Loss:%.4f Accuracy:%.4f' % (epoch,
                                                                                 batch + 1,
                                                                                 len(train_loader),
                                                                                 np.array(losses).mean(),
                                                                                 acc))
                logging.info('Epoch %d [%d/%d] ANN Training Loss:%.4f Accuracy:%.4f' % (epoch,
                                                                                 batch + 1,
                                                                                 len(train_loader),
                                                                                 np.array(losses).mean(),
                                                                                 acc))
                correct = 0.0
                total = 0.0

        # print('weight_noise:',np.array(compute_weight_noise(net.state_dict(), last_state).cpu()))
        weight_noise.append(np.array(compute_weight_noise(net.state_dict(), last_state).cpu()))
        torch.save(net.state_dict(),'last_state.pth')

        plt_tacc.append(acc)  #acc是一个epoch结束后的准确率
        plt_tloss.append(np.array(losses).mean())

        logging.info('Train Loss:%.4f Accuracy:%.4f'%(np.array(losses).mean(),acc))


        net.eval()
        correct = 0.0
        total = 0.0
        losses = []
        with torch.no_grad():
            for batch, (img, label) in enumerate(test_loader):
                img = img.to(device)
                y_one_hot = torch.zeros(batch_size, 10).scatter_(1, label.reshape((label.shape[0], 1)), 1)
                label_seq = y_one_hot.repeat(time_window, 1, 1)

                out = net(img)  # out shape[time_window,batch,10]
                # print(out.T)

                loss = rossum_distance(out, label_seq.to(device))  # loss的大小为【batch】
                #loss=loss_function(out, label_seq.to(device))
                #loss=torch.mean(loss)
                out = torch.sum(out, dim=0)  #[batch,10]
                correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()

                total += out.shape[0]
                losses.append(loss.item())
            acc = correct / total
            if epoch == None:
                print('ANN Validating Accuracy:%.4f' % (acc))
            else:
                plt_eacc.append(acc)
                plt_eloss.append(np.array(losses).mean())
                print('Epoch %d [%d/%d] ANN Validating Loss:%.4f Accuracy:%.4f var:%.4f' % (epoch,
                                                                                   batch + 1,
                                                                                   len(test_loader),
                                                                                   np.array(losses).mean(),
                                                                                   acc,
                                                                                   np.var(losses)))

        logging.info('Validating Loss:%.4f Accuracy:%.4f' % (np.array(losses).mean(),acc))



        if best_acc <= acc:
            best_acc=acc
            torch.save(net, 'best_nminst_snn_hanming.pkl')

    logging.info('trian loss:{}\n weight_noise:{}'.format(plt_tloss,weight_noise))

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