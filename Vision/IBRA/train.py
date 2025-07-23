import torch
import torch.nn as nn
import torchvision
import time
import utils
import os
import model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from kdutils import seed_all
from tqdm import tqdm
import torchvision.transforms as transforms
import data

# torch.autograd.set_detect_anomaly(True)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

dist.init_process_group(backend='nccl', init_method='env://')

traindir = 'your_path'
testdir = 'your_path'

batch_size = 1024 # 每张卡的batchsize 
seed_all(666)
cache_dataset = True
distributed = True
num_epochs = 320


dataset_train, dataset_test, train_sampler, test_sampler = data.load_data(traindir, testdir,
                                                                cache_dataset, distributed)
#dataset_test, test_sampler = load_data(traindir, testdir,
#                                                 cache_dataset, distributed)
print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')



# 根据进程的排名计算局部 GPU ID
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f'cuda:{local_rank}')


train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size,
    sampler=train_sampler,  pin_memory=True, num_workers = 8)

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, pin_memory=True, num_workers = 8)


model = model.spiking_resnet34(T=1).to(device)


model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(model)


loss_fun = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)



start_epoch = 0 
best_accuracy = 0.0  # 用于跟踪最佳验证准确率
best_model_state = None  # 用于保存最佳模型状态

scaler = GradScaler()  # 混合精度

with open(f'results_imagenet_resnet34_seed666_IBRA.txt', 'w') as file:
    for epoch in range(start_epoch, num_epochs):
        
        model.train()  # 将模型设置为训练模式
        t1 = time.time()

        train_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)        
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        # 仅在主进程中显示进度条
        if torch.distributed.get_rank() == 0:
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        else:
            train_loader_tqdm = train_loader

        for i, (data, target) in enumerate(train_loader_tqdm):
            start_time2 = time.time()
            data, target = data.to(device), target.to(device)
            data_load_time = time.time() - start_time2
            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    output = model(data)
                    loss = loss_fun(output, target)

            train_loss += loss.detach()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            train_time = time.time() - start_time2 - data_load_time

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            if torch.distributed.get_rank() == 0:
                train_loader_tqdm.set_postfix({'Loss': (train_loss / total).item(), 'Acc': (100. * correct / total).item()})

        torch.distributed.all_reduce(train_loss)
        torch.distributed.all_reduce(correct)
        torch.distributed.all_reduce(total)

        scheduler.step()
        avg_loss = train_loss / total
        accuracy = 100. * correct / total
        if torch.distributed.get_rank() == 0:
            file.write(f"Epoch {epoch+1}: Average Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
            print(f"Epoch {epoch+1}: Average Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            t2 = time.time()
            file.write(f"训练时间为: {t2-t1}\n")
            print("训练时间为", t2-t1)

        model.eval()
        total2 = torch.tensor(0, device=device)
        correct2 = torch.tensor(0, device=device)
        test_loss = torch.tensor(0.0, device=device)
        test_sampler.set_epoch(epoch)

        if torch.distributed.get_rank() == 0:
            test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
        else:
            test_loader_tqdm = test_loader
        with torch.no_grad():           
        # 测试循环
            test_loss = 0
            correct = 0
            for data, target in test_loader_tqdm:
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += loss_fun(output, target).detach()  # 累加损失
                pred = output.data.max(1, keepdim=True)[1]  # 获取最大概率的索引
                correct2 += pred.eq(target.data.view_as(pred)).sum()
                total2 += target.size(0)
                if torch.distributed.get_rank() == 0:
                    test_loader_tqdm.set_postfix({'Loss': (test_loss / total2).item(), 'Acc': (100. * correct2 / total2).item()})
            torch.distributed.all_reduce(test_loss)
            torch.distributed.all_reduce(correct2)
            torch.distributed.all_reduce(total2)

            test_loss /= total2
            test_accuracy = 100. * correct2 / total2
            t3 = time.time()

        if torch.distributed.get_rank() == 0:
            print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Time: {:.4f}'.format(test_loss.item(), test_accuracy.item(), t3-t2))
            file.write(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%, Time: {t3-t2}\n')
            file.flush()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }
                torch.save(best_model_state, f'best_validation_resnet34_seed666_imagenet_IBRA.pth')
                print('best acc: {:.4f}'.format(best_accuracy.item()))
                file.write('best acc: {:.4f}\n'.format(best_accuracy.item()))
                state = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
            # 其他状态信息
                    }
                torch.save(state, f'last_validation_resnet34_seed666_imagenet_IBRA.pth')