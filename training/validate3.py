import time

import torch
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools

def validate(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        top5 = AverageMeter('Val Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')

    model.eval()

    num_classes = 10  # 假设有10个类别
    conf_matrix = np.zeros((num_classes, num_classes))

    # 模拟混淆矩阵数据
    y_true = []
    y_pred = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output, cfeatures,t,map= model(images)
            _, predicted = torch.max(output, 1)

            # 更新混淆矩阵
            conf_matrix += confusion_matrix(target.cpu().numpy(), predicted.cpu().numpy(),
                                            labels=np.arange(num_classes))

            # 更新混淆矩阵数据
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())


            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 10))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)
        cm_percentage = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True) * 100

        # 定义类别标签
        classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8',
                   'Class 9']

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Percentage)')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # 在图中添加数值标签
        thresh = cm_percentage.max() / 2.
        for i, j in itertools.product(range(cm_percentage.shape[0]), range(cm_percentage.shape[1])):
            plt.text(j, i, '{:.2f}%'.format(cm_percentage[i, j]),
                     horizontalalignment="center",
                     color="white" if cm_percentage[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # 保存混淆矩阵图
        plt.savefig('confusion_matrix.png')

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        with open(args.log_path, 'a') as f1:
            f1.writelines(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                          .format(top1=top1, top5=top5))
        if test:
            tensor_writer.add_scalar('loss/test', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/test', top1.avg, epoch)
            tensor_writer.add_scalar('ACC@5/test', top5.avg, epoch)
        else:
            tensor_writer.add_scalar('loss/val', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/val', top1.avg, epoch)
            tensor_writer.add_scalar('ACC@5/val', top5.avg, epoch)

    return top1.avg