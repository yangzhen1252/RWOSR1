import os
import random
import shutil
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
from loss_reweighting1 import lossb_expect,lossb_expectn,lossb_expecttt

from training.reweighting import weight_learner


def train(train_loader, model, criterion, optimizer, epoch, args, tensor_writer=None):
    ''' TODO write a dict to save previous featrues  check vqvae,
        the size of each feature is 512, os we need a tensor of 1024 * 512
        replace the last one every time
        and a weight with size of 1024,
        replace the last one every time
        TODO init the tensors
    '''

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    feature_dict = {}
    feature_dict.clear()
    feature_dict1 = {}
    feature_dict1.clear()
    feature_dict2 = {}
    feature_dict2.clear()
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        loss3 = 0
        loss5 = 0
        data_time.update(time.time() - end)
        # for key, tensor in feature_dict1.items():
        #         print(f'Tensor: {key}')
        #         print(f'Shape: {tensor}')

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            images, target= images.to(device), target.to(device)
            model = model.to(device)

        output, cfeatures,t,map,map1 = model(images)
        # print(f'target: {target}')
        for j in range(images.size(0)):
                current_feature1 =map[j].detach().view(map[j].size(0), -1)
                category1 = target[j].item()

                if category1 in feature_dict:
                    feature_dict[category1] = (feature_dict[category1] + current_feature1)/2
                    feature_dict1[category1] = feature_dict1[category1] + current_feature1
                else:
                    feature_dict[category1] = current_feature1
                    feature_dict1[category1] = current_feature1

        for j in range(images.size(0)):
            current_feature = t[j].detach().view(map[j].size(0), -1)
            category = target[j].item()

            if category in feature_dict2:
                feature_dict2[category] = feature_dict2[category] + current_feature

            else:
                feature_dict2[category] = current_feature
        for j in range(images.size(0)):
            current_feature3 = map[j].detach().view(map[j].size(0), -1)
            categoryt = target[j].item()

            if categoryt in feature_dict:
                loss3=lossb_expecttt(feature_dict[categoryt],current_feature3,128)
                loss3 += loss3
        for mm in range(images.size(0)):
            current_feature77 = map[mm].detach().view(map[mm].size(0), -1)
            current_feature88 = map1[mm].detach().view(map1[mm].size(0), -1)
            loss5 = lossb_expecttt(current_feature88, current_feature77, 128)
            loss5 += loss5
       # print(f'loss3: {loss3}')



        # for key, tensor in feature_dict2.items():
        #         print(f'Tensor: {key}')
        #         print(f'Shape: {tensor.shape}')
        #         print()  # 打印空行以区分不同张量
        # 继续训练模型...
        # 假设feature_dict是一个保存特征图的字典
        if len(feature_dict) >= 10:
            loss1 = lossb_expect(feature_dict,128)
            feature_dict={}
            feature_dict.clear()
           # print(f'loss1: {loss1}')

        else:
            # 字典中的元素个数不大于2，不执行其他操作
            loss1 = 0






        pre_features =  model.pre_features
        pre_weight1 = model.pre_weight1
        # print(f'loss1: {loss1}')
        # print(f'loss3: {loss3}')
        if epoch >= args.epochp:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

        model.pre_features.data.copy_(pre_features)
        model.pre_weight1.data.copy_(pre_weight1)

       # print(f'loss: {criterion(output, target).view(1, -1).mm(weight1).view(1)}')
        loss = criterion(output, target).view(1, -1).mm(weight1).view(1)+loss1*10+loss3/10+loss5/10
        #loss = criterion(output, target).view(1, -1).mm(weight1).view(1)
        #print(f'loss: {loss}')

        acc1, acc5 = accuracy(output, target, topk=(1, 10))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        method_name = args.log_path.split('/')[-2]
        if i % args.print_freq == 0:
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)
    torch.save(feature_dict, "resultslocal/PACS/feature_dict_%d.pt" % (epoch))
    torch.save(feature_dict1, "resultslocal/PACS/feature_dict1_%d.pt" % (epoch))
    torch.save(feature_dict2, "resultslocal/PACS/feature_dict2_%d.pt" % (epoch))
    tensor_writer.add_scalar('loss/train', losses.avg, epoch)
    tensor_writer.add_scalar('ACC@1/train', top1.avg, epoch)
    tensor_writer.add_scalar('ACC@5/train', top5.avg, epoch)

