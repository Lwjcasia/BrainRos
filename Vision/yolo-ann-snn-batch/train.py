import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from torch.nn import DataParallel
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed


wdir = 'weights' + os.sep  # weights dir 权重目录
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train(hyp):
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    # Image Sizes
   #  设置网格大小为32像素。这可能与某些类型的神经网络，如YOLO，相关，其中图像被划分为大小为 gs x gs 的格子。
    gs = 32  # (pixels) grid size
    # 确保 imgsz_min 是 gs 的倍数。这通常是为了确保图像大小与神经网络的某些结构特性（如卷积层的步长或池化层的大小）兼容
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    #如果 imgsz_min 和 imgsz_max 不相等，它将设置 opt.multi_scale 为 True。
    # multi_scale 可能是一个标志，指示是否在多个尺度上训练模型，这对于某些对象检测任务是有益的。
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)

    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        #计算了最小和最大的网格尺寸
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        # 修正 imgsz_min 和 imgsz_max，确保它们是网格大小 gs 的倍数  感觉有点多此一举哦，前面已经assert过了
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    # 表示当前用于训练的图像尺寸
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()  # 设置随机种子
    data_dict = parse_data_cfg(data)
    
    train_path = data_dict['train']  # 训练路径
    test_path = data_dict['valid']   # 测试路径
    # 如果 opt.single_cls 为 True（这意味着只有一个类别），nc 将被设置为1。
    # 否则，它将从 data_dict 中提取类别数
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes 
 # 这里，'cls' 的值被乘以 nc / 80。这个操作可能是基于一个预先在COCO数据集上训练的模型（COCO有80个类别），现在它正在调整到当前数据集的类别数。
 # 如果您的数据集有与COCO不同的类别数，这会帮助确保类别损失（可能是交叉熵损失）的权重与数据集的类别数相匹配。
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
    # 对于上面找到的每个文件 f，这行代码将其删除。这可能是为了在新的训练开始之前清理旧的输出或结果。
        os.remove(f)

    # Initialize model
    #  初始化模型 ， 有无timestep 变成不同的模型

    # 更改成用多个gpu
    #device_ids = [0, 1]
    if opt.timesteps is None:
        #model = Darknet(cfg).to(device)
        model = Darknet(cfg).cuda()
        # from torch.nn.parallel import DistributedDataParallel as DDP
        # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
       # print(next(model.parameters()).device)
       #torch.nn.DataParallel(model)
       # model=torch.nn.parallel.DistributedDataParallel(model)
       # print(next(model.parameters()).device)
       # model = Darknet(cfg)
       # model = DataParallel(model, device_ids=device_ids)
       # model = model.to('cuda:{}'.format(device_ids[0]))
    else:
        model = Darknet_Q(cfg, timesteps=opt.timesteps).to(device)
        #model = Darknet_Q(cfg, timesteps=opt.timesteps).cuda()
        #torch.nn.DataParallel(model, device_ids=[0, 1])
    # Optimizer
    # 该代码将模型中的参数分为三个不同的组：偏置项、卷积层权重和其他所有参数。
    # 这样做的目的可能是为了在接下来的优化过程中为这三组参数设置不同的优化选项，例如不同的学习率或权重衰减。
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    # 对不同组不同优化
    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # 打印出每个参数组中的参数数量。这有助于验证和理解模型中的参数是如何被分组的。
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    # 删除了之前定义的三个参数组，释放它们占用的内存。由于这些参数组已经被添加到优化器中，所以我们不再需要它们的独立引用
    del pg0, pg1, pg2

    start_epoch = 0   # 开始的周期
    best_fitness = 0.0  # 最佳性能
    # 尝试从远程服务器下载预训练的权重文件，如果它们在本地没有找到。这对于那些使用预训练模型开始他们的训练任务的用户非常有用，因为他们不必手动下载权重。
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format 检查权重文件是否是pytorch 格式

        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        # 这行代码加载权重文件，并将其存储在 ckpt 变量中。map_location=device 确保权重被加载到正确的计算设备上（CPU或GPU）。
        ckpt = torch.load(weights, map_location=device)

        # load model 加载模型权重
        try:
            # 过滤 ckpt['model'] 中的键值对，只保留那些与当前模型 model 的权重形状匹配的权重。
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            # 使用 load_state_dict 方法将过滤后的权重加载到模型中。strict=False 指示函数在模型和权重之间的大小不匹配时不要引发错误。
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            # 创建一个错误消息，告诉用户所提供的权重与配置文件不兼容。
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer  ckpt  --> checkpoint
        # 检查检查点文件中是否包含优化器的状态
        if ckpt['optimizer'] is not None:
            # 使用 load_state_dict 方法从检查点加载优化器的状态。这非常有用，
            # 尤其是当你想要从某个断点继续训练时。加载优化器的状态可以确保训练继续进行，而不是从头开始
            optimizer.load_state_dict(ckpt['optimizer'])
            # 从检查点加载最佳适应度值。这可能是在训练过程中记录的模型的最佳性能指标
            best_fitness = ckpt['best_fitness']

        # load results
        # 检查检查点文件中是否包含之前的训练结果
        # 将检查点中的训练结果写入文件。这通常是为了记录训练过程中的某些指标，如损失、准确率等。
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        # 我感觉就是你下载的权重有可能是别人已经训练了几轮的
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt
    # 这部分代码为那些想使用Darknet格式的预训练权重（如从YOLO的官方网站下载的权重）的用户提供了支持。它确保这些权重可以被正确地加载到PyTorch模型中，
    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

# 当我们说“冻结”一层时，我们意味着在训练过程中不更新该层的权重。这通常在迁移学习中使用，
# 当我们想保留预训练模型的某些部分不变，而只调整或训练模型的其他部分时
    if opt.freeze_layers:
        # 输出层，并获得他们的索引
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)]
        # 这行代码获取除输出层及其前一层之外的所有层的索引。冻结除了输出层和输出层前一层之外的其他所有层
        freeze_layer_indices = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        # 冻结
        for idx in freeze_layer_indices:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # Mixed precision training https://github.com/NVIDIA/apex
    # 这是一种加速深度学习训练的技术，同时仍然保持了模型性能。混合精度利用了较低精度（如半精度或float16）的算术来加速训练，
    # 但在关键部分使用较高精度（如float32）以保持数值稳定性。
    if mixed_precision:
        # opt_level='O1': 这指定了混合精度的优化级别。在O1级别，大多数的操作都会使用半精度（float16）进行，
        # 但一些关键操作会使用单精度（float32）以保持数值稳定性。
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 具体是使用了余弦退火策略（Cosine Annealing）来动态调整学习率。学习率调度是深度学习中的一个常用技巧，用于在训练过程中调整学习率。
    # 通过适当地降低学习率，可以帮助模型在训练晚期更好地收敛
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    # 使用PyTorch的lr_scheduler模块创建了一个学习率调度器。LambdaLR是一种基于用户提供的函数（在这里是lf）来调整学习率的调度器。
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # 设置调度器的last_epoch属性为start_epoch - 1。这是为了确保调度器从正确的周期开始调整学习率，特别是当从检查点恢复训练时。
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    # if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
    #     dist.init_process_group(backend='nccl',  # 'distributed backend'
    #                             init_method='tcp://127.0.0.1:9999',  # distributed training init method
    #                             world_size=1,  # number of nodes for distributed training
    #                             rank=0)  # distributed training node rank
    #     model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False) # original True
    #     model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    #  看看是VOC数据集还是COCO数据集，然后加载数据
    if data.split('/')[1].startswith('voc'):
        LoadImagesAndLabels = LoadVOCImagesAndLabels
    else:
        LoadImagesAndLabels = LoadCOCOImagesAndLabels
    #  这样的话可以得到图像和标签
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                augment=True,
                                hyp=hyp,  # augmentation hyperparameters
                                rect = opt.rect,  # rectangular training
                                cache_images=opt.cache_images,
                                single_cls=opt.single_cls)
    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    # from torch.utils.data.distributed import DistributedSampler
    # sampler = DistributedSampler(dataset) # 这个sampler会自动分配数据到各个gpu上
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=batch_size,
    #                                          sampler = sampler,
    #                                          num_workers=nw,
    #                                         # shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
    #                                          pin_memory=True,
    #                                          collate_fn=dataset.collate_fn)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    # Testloader
    #  跟刚刚一样，只不过用了test_path，合并写成一条了
    # dataset2 = LoadImagesAndLabels(test_path, imgsz_test, batch_size,hyp=hyp,rect=opt.rect,cache_images=opt.cache_images,single_cls=opt.single_cls)
    # sampler2 = DistributedSampler(dataset2)
    # testloader = torch.utils.data.DataLoader(dataset2,
    #                                          batch_size=batch_size,
    #                                          sampler=sampler2,
    #                                          num_workers=nw,
    #                                          pin_memory=True,
    #                                          collate_fn=dataset.collate_fn)
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    # Model parameters
    model.nc = nc  # attach number of classes to model 类别数
    model.hyp = hyp  # attach hyperparameters to model 超参数
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou) giou损失率
    model.class_weights = labels_to_class_weights(dataset.labels, nc).cuda()  # attach class weights 类别权重

    # Model EMA
    # 为模型创建一个指数移动平均（EMA）对象。这可以帮助模型在训练过程中平滑预测。
    ema = torch_utils.ModelEMA(model)

    # Start training
    # 设置与训练相关的一些初始参数，如批处理的数量、预热迭代次数和每个类别的mAP（平均准确率）。
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    # 初始化结果元组和开始时间。
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()  # 训练模式
        # 当训练图像的所有类个数不相同时,我们可以更改类权重, 即而达到更改图像权重的目的.然后根据图像权重新采集数据，这在图像类别不均衡的数据下尤其重要
        # Update image weights (optional)
        if dataset.image_weights:  # 图像权重开启的话
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights  基于模型的当前类权重和每个类的mAP
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w) # 根据类权重和数据集的标签计算图像权重
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx 根据权重抽取图像
        # 初始化一个包含4个零的张量，用于存储平均损失
        mloss = torch.zeros(4).cuda()  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar 使用tqdm库创建一个进度条，用于在训练时显示进度
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # if i > 1:
            #     break
            ni = i + nb * epoch  # number integrated batches (since train start)  从训练开始到现在的总批次数
            # 将图像转移到设备（例如GPU），将数据类型转换为浮点数，并将值从[0,255]范围归一化到[0,1]范围。
            imgs = imgs.cuda().float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0  
            # 将目标（如边界框和类标签）转移到设备（例如GPU）
            targets = targets.cuda()
            # 接下来的部分涉及“烧入”（burn-in）过程，它是深度学习训练早期的一个常见策略，用于慢慢地增加学习率，以防止模型在训练初期发生不稳定。
            # Burn-in
            if ni <= n_burn:
                # 调整学习率、权重衰减和动量等优化器参数的步骤
                xi = [0, n_burn]  # x interp 代表烧入期的起始和结束批次数，用于随后的线性插值函数 
                # 使用线性插值调整模型的gr参数。在烧入期开始时，gr为0.0，烧入期结束时，gr为1.0。
                # gr可能代表GIoU（Generalized Intersection over Union）损失与对象损失之间的权重比。
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                # 这行代码调整累积梯度更新的频率。在烧入期开始时，每个批次都更新梯度，但随着时间的推移，更新可能会减少，特别是当批次大小较大时
                accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                # 循环遍历优化器的所有参数组。这通常包括主要权重、偏置和BatchNorm层的参数,在一个batch内的
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    # 对于偏置项（当j==2时），学习率从0.1线性增加到其初始值。对于其他参数，学习率从0.0线性增加到其初始值
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    # 对于BatchNorm层的参数（当j==1时），权重衰减从0.0线性增加到其设定值
                    x['weight_decay'] = np.interp(ni, xi, [0.0, hyp['weight_decay'] if j == 1 else 0.0])
                    # 如果参数组具有动量，则其值从0.9线性增加到预定值。动量是一种加速SGD（随机梯度下降）的方法，它考虑了过去的梯度，以更平稳地更新参数
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-Scale  检查是否启用了多尺度训练
            if opt.multi_scale:
                # 确保每accumulate批次调整一次图像大小
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    # 图像大小是在grid_min和grid_max之间随机选择的一个值，然后乘以gs（可能是一个网格大小因子）。这确保了新的img_size是gs的倍数。
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                # 计算尺度因子sf。这是新的img_size与当前批次图像的最大维度之间的比例。imgs.shape[2:]表示图像的高度和宽度（通常，图像的形状是[批次大小, 通道数, 高度, 宽度]）。
                sf = img_size / max(imgs.shape[2:])  # scale factor
                # 不相等，我们才需要调整图像的大小
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            # 累积（accumulate）是一个用于模拟更大批次大小的技术，特别是当真正的大批次不能直接在硬件（如GPU）上使用时。这种技术通常被称为“累积梯度”或“梯度累积”
            # accumulate是梯度更新的频率
            # 实际更新模型参数的批次大小等于batch_size * accumulate。
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            # 印每批次的平均损失、使用的GPU内存、当前的进度等，并可选择地为TensorBoard绘制图像
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            # 这里更新了平均损失mloss。基本思想是将当前批次的损失loss_items与之前批次的总损失相加，然后除以批次数来计算新的平均损失
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)
            # 初次启动训练时，开发者可能希望立即查看模型处理的图像和标签，以确保数据加载和预处理正确，并且模型开始从数据中学习。
            # Plot
            if ni < 1:
                f = 'train_batch%g.jpg' % i  # filename
                res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        # 更新学习率调度器。随着训练的进行，学习率可能需要按照预定的策略进行调整
        scheduler.step()

        # Process epoch results
        # 更新模型的指数移动平均（EMA）。EMA是一种常用的技术，用于平滑模型参数的变化
        ema.update_attr(model)
        # 保存模型、评估模型性能、更新学习率和记录训练结果
        final_epoch = epoch + 1 == epochs
        # 写入结果
        if not opt.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            
            t1_ann = time.time()
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      imgsz=imgsz_test,
                                      model=ema.ema,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      dataloader=testloader,
                                      multi_label=ni > n_burn)
            t2_ann = time.time()
            print('ann时间为：',t2_ann-t1_ann)
        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi


        # Save model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    
    print('best_fitness is: ', best_fitness)

    # 权重模型写入
    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    if not opt.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    # dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=32)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-mp2conv-mp1none-lk2relu-up2tconv.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='/disk1/ybh/data/coco.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames .txt/.pt to _name.txt/_name.pt if supplied')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
    parser.add_argument('--timesteps', type=int, default=None, help='timesteps ann-to-snn will use')
    parser.add_argument('--gpu_id', type=str, default='0,1')
    opt = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    last = wdir + f"last_{opt.name}.pt" if len(opt.name) > 0 else last  # last.pt maybe renamed 指向保存的模型权重
    best = wdir + f"best_{opt.name}.pt" if len(opt.name) > 0 else best  # best.pt maybe renamed 最佳的模型权重
    results_file = f"results/train_{opt.name}.txt" if len(opt.name) > 0 else results_file  # results.txt maybe renamed  训练结果的文件
    opt.weights = last if opt.resume and not opt.weights else opt.weights # ?? 如果用户选择继续之前的训练（opt.resume 为 True）并且没有提供权重文件路径（not opt.weights），则 opt.weights 将被设置为 last。否则，它将保持原样

    # check_git_status()
    opt.cfg = check_file(opt.cfg)  # check file 用于定义神经网络模型的结构和参数
    opt.data = check_file(opt.data)  # check file  数据
    #这行代码确保 opt.img_size 列表的长度为3。
   # 如果 opt.img_size 的长度小于3，它会用最后一个元素来填充列表，直到长度为3。
    #这可能是为了确保有三个图像尺寸（可能对应于最小、最大和测试尺寸）。
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    print(opt)
    
    # device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
   # device = torch.device('cuda:{}'.format(opt.device))

    # 当使用单个GPU下面这行可以注释掉将上面那个打开，这是为了保证其他一些数据在cuda0上运行
   # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

# 可能是TensorBoard的写入器对象，用于记录训练过程中的数据，以便后续在TensorBoard中进行可视化。将其初始化为None意味着在这个时点，没有TensorBoard日志被创建或写入。
    tb_writer = None
    # print(opt.evolve)  False
    if not opt.evolve:  # Train normally
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')  # 可以登入这个网站查看训练可视化
        tb_writer = SummaryWriter(comment=opt.name) # SummaryWriter 是 PyTorch 的 TensorBoard 工具的一部分，用于将数据写入 TensorBoard 格式，以便后续进行可视化
        train(hyp)  # train normally

    else:  # Evolve hyperparameters (optional)
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(hyp.copy())

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
