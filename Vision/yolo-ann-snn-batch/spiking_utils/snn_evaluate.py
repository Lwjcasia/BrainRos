import json

from torch.utils.data import DataLoader

from models import *
from spiking_utils import spike_tensor
from spiking_utils.spike_tensor import SpikeTensor
from utils.datasets import *
from utils.utils import *


def ann_evaluate(opt,
                 data,
                 ann,
                 partial_ann,
                 batch_size=16,
                 imgsz=416,
                 conf_thres=0.001,
                 iou_thres=0.6,  # for nms
                 save_json=False,
                 single_cls=False,
                 augment=False,
                 dataloader=None,
                 multi_label=True):
    verbose = opt.task == 'test'
    device = torch_utils.select_device(opt.device, batch_size=batch_size)
    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    ann.eval()
    partial_ann.eval()
    _ = ann(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            output1, output2 = partial_ann(imgs)  # two branches
            # post-processing: conv, yolo
            output1 = ann.module_list[14](output1)
            output2 = ann.module_list[-2](output2)
            yolo_outputs, out = [], []
            yolo_outputs.append(ann.module_list[15](output1, out))
            yolo_outputs.append(ann.module_list[-1](output2, out))
            inf_out, _ = zip(*yolo_outputs)  # inference output, training output
            inf_out = torch.cat(inf_out, 1)  # cat yolo outputs
            if augment:  # de-augment results
                inf_out = torch.split(inf_out, nb, dim=0)
                inf_out[1][..., :4] /= s[0]  # scale
                img_size = imgs.shape[-2:]  # height, width
                inf_out[1][..., 0] = img_size[1] - inf_out[1][..., 0]  # flip lr
                inf_out[2][..., :4] /= s[1]  # scale
                inf_out = torch.cat(inf_out, 1)
            t0 += torch_utils.time_synchronized() - t

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


def snn_evaluate(opt,
                 data,
                 ann,
                 snn,
                 timesteps=16,
                 batch_size=16,
                 imgsz=416,
                 conf_thres=0.001,
                 iou_thres=0.6,  # for nms
                 save_json=False,
                 single_cls=False,
                 augment=False,
                 dataloader=None,
                 multi_label=True):
    
    if opt.data.split('/')[-1].startswith('voc'):
        LoadImagesAndLabels = LoadVOCImagesAndLabels
    elif opt.data.split('/')[-1].startswith('coco'):
        LoadImagesAndLabels = LoadCOCOImagesAndLabels
    verbose = opt.task == 'test'
    # device = torch_utils.select_device(opt.device, batch_size=batch_size)
    device = torch.device('cuda:{}'.format(opt.device))
    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    # iouv和niou用于mAP计算
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    ann.eval()
    snn.eval()
    ann.to(device)
    snn.to(device)
    # 为ANN模型运行一次前向传递（如果在GPU上）以进行初始化
    _ = ann(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    # 初始化COCO类映射、统计变量和其他数据结构
    coco91class = coco80_to_coco91_class() # converts 80-index (val2014) to 91-index (paper)
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    total_firing_ratios = []

    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)
        # 使用时间步复制数据，然后将其转换为SpikeTensor。
        replica_data = torch.cat([imgs for _ in range(timesteps)], 0)  # replica for input(first) layer
        data = SpikeTensor(replica_data, timesteps, scale_factor=1)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()   # 现在的时间
            spike_tensor.firing_ratio_record = True
            output_snn1, output_snn2 = snn(data)  # two branches
            spike_tensor.firing_ratio_record = False
            output_ann1 = output_snn1.to_float()  # spike to real-value
            output_ann2 = output_snn2.to_float()
            # post-processing: conv, yolo
            # 进入YOLO层前需要转化成连续值输入
            output_ann1 = ann.module_list[14](output_ann1)
            output_ann2 = ann.module_list[-2](output_ann2)
            yolo_outputs, out = [], []
            yolo_outputs.append(ann.module_list[15](output_ann1, out))
            yolo_outputs.append(ann.module_list[-1](output_ann2, out))
            # 将两个YOLO输出连接在一起
            inf_out, _ = zip(*yolo_outputs)  # inference output, training output
            inf_out = torch.cat(inf_out, 1)  # cat yolo outputs

            if augment:  # de-augment results  数据增强
                inf_out = torch.split(inf_out, nb, dim=0)
                inf_out[1][..., :4] /= s[0]  # scale
                img_size = imgs.shape[-2:]  # height, width
                inf_out[1][..., 0] = img_size[1] - inf_out[1][..., 0]  # flip lr
                inf_out[2][..., :4] /= s[1]  # scale
                inf_out = torch.cat(inf_out, 1)
            # 更新模型前向传播的总执行时间
            t0 += torch_utils.time_synchronized() - t

            # Run NMS
            t = torch_utils.time_synchronized()
            # 使用NMS对模型输出进行后处理，以删除冗余和低置信度的预测框，并更新NMS的执行时间。
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        # 对于给定批次中的每个图像，output包含该图像的预测边界框
        for si, pred in enumerate(output):
            # 真实标签 其中每行代表一个标签（图像索引，类别，边界框坐标）。这里提取与当前图像相关的所有标签。
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            # 目标类
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # seen变量记录了到目前为止处理过的图像数
            seen += 1
            # 如果当前图像没有任何预测，但有真实标签，就添加一个空的统计记录
            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            # 将预测的边界框坐标修剪到图像的边界内
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            # 如果设置了save_json标志，此段代码将预测的边界框添加到一个JSON字典中
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    #  主要就是添加这些
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})
            # 这段代码的目的是为每个预测边界框分配一个“正确”或“错误”的标签，基于预测与实际目标的IoU（交并比）。这对于后续的评估和mAP（平均准确率）计算是必需的
            # Assign all predictions as incorrect
            # 初始化一个布尔张量，其长度与预测数量相同，并将所有元素设置为False
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                # 用于跟踪已经被检测到的真实目标
                detected = []  # target indices
                tcls_tensor = labels[:, 0] # 提取真实标签的类别

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh # 将目标的边界框从中心宽高格式转换为左上-右下格式，并按图像大小进行缩放

                # Per target class 遍历真实标签中的每个唯一类别
                for cls in torch.unique(tcls_tensor):
                    # 获取与当前类别匹配的目标和预测索引
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    # 如果有与当前类别匹配的预测，则计算每个预测与所有目标的IoU，并为每个预测分配最佳匹配的目标
                    if pi.shape[0]:
                        # Prediction to target ious
                        # 使用box_iou函数计算预测的边界框与真实的边界框之间的交并比(IoU)。max(1)确保我们只获得与每个预测最匹配的真实边界框的IoU和其索引
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        # 对于那些IoU超过阈值iouv[0]的预测，我们继续处理它们
                        # 这里代码应该是有些冗余的,j这个循环就保证了Ious[j] > iouv
                    
                        for j in (ious > iouv[0]).nonzero():
                            # 从与当前预测最匹配的真实边界框中获取目标的索引
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                # 如果目标还没有被检测到
                                detected.append(d)
                                # 对于当前的预测，检查其IoU是否超过了阈值iouv。如果是，则将此预测标记为正确
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                # 检查是否所有的目标都已经被检测
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # 对于每个图像，将其正确的预测、预测的置信度、预测的类别和真实的类别添加到stats列表中。                                   
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f)  # predictions

        total_firing_ratios.append([_.mean().item() for _ in spike_tensor.firing_ratios])
        spike_tensor.firing_ratios = []

    # Compute statistics
    # 这一行是对收集到的stats进行整合。在前面的代码中，每次处理一个图像时，都会为这个图像收集一个统计信息，存放在stats列表中。
    # 这里，我们将这些统计信息整合到一个列表中，其中每个元素是一个numpy数组。
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        # 使用ap_per_class函数计算每个类别的准确率（P）、召回率（R）、平均准确率（AP）和F1分数。这个函数会为每个类别返回这些指标
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        # 这个条件判断是检查是否为每个IoU阈值都计算了指标。如果是，那么只选择AP@0.5的值，并为所有IoU阈值计算平均值。
        # 这是因为在某些评估中，我们可能只对一个特定的IoU阈值感兴趣，而在其他评估中，我们可能对一系列的IoU阈值都感兴趣。
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        # 使用np.bincount计算每个类别的目标数量。这可以帮助我们了解每个类别有多少个标记的实例，从而对性能指标有更好的理解
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class 每个类的结果
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds 速度 
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

# 这一段代码的目的是利用pycocotools（官方COCO评估工具）来计算COCO数据集上的mAP。
    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        # 从dataloader提取图像的ID。这些ID将用于后续的COCO评估
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        # 将jdict中的预测结果保存到results.json文件中
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            # 使用pycocotools的COCO类初始化COCO ground truth API。这个API将用于获取真实标注数据。
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api
            # 初始化COCO评估对象，用于比较真实标注和预测结果
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            # 设置要评估的图像ID
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map  # 初始化为这个是可能在验证集中有未出现的类别
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    total_firing_ratios = np.mean(total_firing_ratios, 0)
    mean_firing_ratio = total_firing_ratios.mean()
    print(f"Mean Firing ratios {mean_firing_ratio}, Firing ratios: {total_firing_ratios}")
    
    # 遍历SNN模型的所有层。如果某一层有mem_potential属性（代表神经元的膜电位），将其设置为None以释放内存
    for layer in snn.modules():
        if hasattr(layer, 'mem_potential'):
            layer.mem_potential = None
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps, total_firing_ratios
