import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
from loss_reweighting1 import predict_class,predict_class1,predict_class2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def validate(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        top2 = AverageMeter('Acc@2', ':6.2f')
        top6 = AverageMeter('Acc@6', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5,top2, top6],
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
    score_dict= {}
    score_dict.clear()
    nn=0
    features = []
    labels = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            gcfeatures = torch.load('results/PACS/feature_dict1_183.pt')
            # for key, tensor in gcfeatures.items():
            #         print(f'Tensor: {key}')
            #         print(f'Shape: {tensor.shape}')
            gcfeatures1 = torch.load('results/PACS/feature_dict2_183.pt')

            output, cfeatures,t,map = model(images)
            features.append(cfeatures.cpu())
            labels.append(target.cpu())


            score,scores=predict_class(map,gcfeatures,128)




            score2=predict_class2(cfeatures,gcfeatures1,128)

            max_valuett, max_index1t = torch.max(score, dim=0)


            cc=target[0].item()


            if cc in score_dict:

                a=score_dict[cc]
                b=torch.tensor([score[cc]])
                score_dict[cc] = torch.cat([a,b])


            else:
                score_dict[cc] = torch.tensor([score[cc]])

            # 创建全零张量，并按照最大值索引设置对应位置为1



            # score11 = predict_class(cfeatures, 6)
            # print(f'score: {score}')
            # print(f'score2: {score2}')

            #print(score.cuda(args.gpu, non_blocking=True)*output.squeeze())
            out1=score.unsqueeze(0).cuda(args.gpu, non_blocking=True)

            # print(f'target: {target}')
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 10))
            acc2, acc6= accuracy(out1, target, topk=(1, 10))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            top2.update(acc2[0], images.size(0))
            top6.update(acc6[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            max_value1, max_index1 = torch.max(output, dim=1)
            # 创建全零张量，并按照最大值索引设置对应位置为1
            result1 = torch.zeros_like(output)
            result1[0][max_index1] = 1
            # print(f'result1: {result1}')

            max_value2, max_index2 = torch.max(score2.unsqueeze(0), dim=1)
            # 创建全零张量，并按照最大值索引设置对应位置为1
            result2 = torch.zeros_like(score2.unsqueeze(0))
            result2[0][max_index2] = 1
            # print(f'result2: {result2}')

            max_value, max_index = torch.max(score.unsqueeze(0), dim=1)

            # 创建全零张量，并按照最大值索引设置对应位置为1
            result = torch.zeros_like(score.unsqueeze(0))
            result[0][max_index] = 1
            # print(f'result: {result}')
            open1 = result.cuda(args.gpu, non_blocking=True) * result1
            # print(f'open: {open1}')
            is_all_zero = torch.all(open1 == 0)
            if is_all_zero==False:
                nn=nn+1
            # print(is_all_zero)
            #print(nn)

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)
        features = torch.cat(features, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()

        # 使用 t-SNE 算法将特征降维到 2D 空间
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)

        # 获取类别标签的唯一值
        unique_labels = np.unique(labels)

        # 绘制分布图
        plt.figure(figsize=(10, 8))
        # 定义不同类别所使用的颜色
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray'] # 可根据类别数量自定义颜色

        for i, label in enumerate(unique_labels):
            # 找到属于当前类别的样本的索引
            indices = np.where(labels == label)
            # 取出属于当前类别的样本的特征
            features_class = features_2d[indices]
            # 绘制当前类别的散点图，并使用对应的颜色
            plt.scatter(features_class[:, 0], features_class[:, 1], label=label, color=colors[i])

        plt.title('Feature Distribution by Class')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig('feature_distribution.jpg')  # 保存分布图为 JPG 格式的图像
        plt.show()
        for key, tensor in score_dict.items():
                print(f'Tensor: {key}')
                print(f'Shape: {tensor}')
        for key, tensor in score_dict.items():
            # 计算张量的均值和标准差
            mean = torch.mean(tensor)
            std = torch.std(tensor)

            # 将均值减去2倍标准差，并重新赋值给对应键
            score_dict[key] = mean-2*std
        for key, tensor in score_dict.items():
                print(f'Tensor: {key}')
                print(f'Shape: {tensor}')
        torch.save(score_dict, 'results/PACS/score_dict.pt')
        for key, tensor in score_dict.items():
                print(f'Tensor: {key}')
                print(f'Shape: {tensor.shape}')
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print(' * Acc@2 {top2.avg:.3f} Acc@6 {top6.avg:.3f}'
              .format(top2=top2, top6=top6))
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

def validate1(val_loader, model, criterion,  args=None, ):

    model.eval()
    score_dict= {}
    score_dict.clear()
    nn=0
    features = []
    labels = []
    with torch.no_grad():

        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            gcfeatures = torch.load('resultsFU/PACS/feature_dict1_196.pt')

           # gcfeatures1 = torch.load('results/PACS/feature_dict2_183.pt')

            output, cfeatures,t,map = model(images)
            features.append(cfeatures.cpu())
            labels.append(target.cpu())


            score,scores=predict_class(map,gcfeatures,128)




          #  score2=predict_class2(cfeatures,gcfeatures1,128)




            cc=target[0].item()


            if cc in score_dict:

                a=score_dict[cc]
                b=torch.tensor([score[cc]])
                score_dict[cc] = torch.cat([a,b])


            else:
                score_dict[cc] = torch.tensor([score[cc]])


            out1=score.unsqueeze(0).cuda(args.gpu, non_blocking=True)

            loss = criterion(output, target)
            print(f'output: {output}')
            print(f'out1: {out1}')
            acc1, acc5 = accuracy(output, target, topk=(1, 10))
            acc2, acc6= accuracy(out1, target, topk=(1, 10))




            # max_value1, max_index1 = torch.max(output, dim=1)
            # # 创建全零张量，并按照最大值索引设置对应位置为1
            # result1 = torch.zeros_like(output)
            # result1[0][max_index1] = 1
            # # print(f'result1: {result1}')
            #
            # max_value2, max_index2 = torch.max(score2.unsqueeze(0), dim=1)
            # # 创建全零张量，并按照最大值索引设置对应位置为1
            # result2 = torch.zeros_like(score2.unsqueeze(0))
            # result2[0][max_index2] = 1
            # # print(f'result2: {result2}')
            #
            # max_value, max_index = torch.max(score.unsqueeze(0), dim=1)
            #
            # # 创建全零张量，并按照最大值索引设置对应位置为1
            # result = torch.zeros_like(score.unsqueeze(0))
            # result[0][max_index] = 1
            # # print(f'result: {result}')
            # open1 = result.cuda(args.gpu, non_blocking=True) * result1
            # # print(f'open: {open1}')
            # is_all_zero = torch.all(open1 == 0)
            # if is_all_zero==False:
            #     nn=nn+1
            # # print(is_all_zero)
            # #print(nn)



    return 1


