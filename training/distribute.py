import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
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
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.metrics import confusion_matrix
def validate(val_loader, model,  epoch=0, test=True, args=None, tensor_writer=None):

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

        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, cfeatures,t,map = model(images)
            features.append(cfeatures.cpu())
            labels.append(target.cpu())


            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)
        features = torch.cat(features, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()


        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)


        unique_labels = np.unique(labels)


        plt.figure(figsize=(10, 8))

        colors = ['red', 'green', 'blue', 'yellow', 'orange','pink','cyan','saddlebrown','gray','purple','black','brown','magenta'] # 可根据类别数量自定义颜色
        #colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'cyan', 'saddlebrown','gray','purple']
        for i, label in enumerate(unique_labels):

            indices = np.where(labels == label)

            features_class = features_2d[indices]

            plt.scatter(features_class[:, 0], features_class[:, 1], label=label, color=colors[i])

        plt.title('Feature Distribution by Class')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig('feature_distributionun111.jpg')
        plt.show()


def validate1(val_loader, model, epoch=0, test=True, args=None, tensor_writer=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        top2 = AverageMeter('Acc@2', ':6.2f')
        top6 = AverageMeter('Acc@6', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5, top2, top6],
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
    score_dict = {}
    score_dict.clear()
    nn = 0
    features = []
    labels = []
    with torch.no_grad():
        all_pred = []
        all_true = []
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, cfeatures, t, map = model(images)
            features.append(cfeatures.cpu())
            labels.append(target.cpu())
            _, predicted = torch.max(output, 1)
            all_pred.extend(predicted.tolist())
            all_true.extend(target.tolist())
            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)


        cm = confusion_matrix(all_true, all_pred)
        cm_sum = np.sum(cm, axis=1, keepdims=True)


        cm_percent = np.round((cm / cm_sum) * 100, 1)

        labels = ['bear', 'bird', 'cat','cow','dog','elephant','horse','monkey','rat','sheep']  # 根据你的分类类别数量设置


        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)





        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('confusion_matrixa.png',dpi=500)

        plt.show()
def evaluation(net2, testloader, outloader,args=None):
    net2.eval()
    correct, total, n = 0, 0, 0
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open = [], [], [], []
    open_labels = torch.zeros(50000)
    probs = torch.zeros(50000)
    score_dict = {}
    score_dict.clear()

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            with torch.set_grad_enabled(False):
                logits, cfeatures,t,map = net2(data)
                gcfeatures = torch.load('resultsFU/PACS/feature_dict1_196.pt')

                score, scores = predict_class(map, gcfeatures, 48)

                #  score2=predict_class2(cfeatures,gcfeatures1,128)

                cc = labels[0].item()

                if cc in score_dict:

                    a = score_dict[cc]
                    b = torch.tensor([score[cc]])
                    score_dict[cc] = torch.cat([a, b])


                else:
                    score_dict[cc] = torch.tensor([score[cc]])

                out1 = score.unsqueeze(0).cuda(args.gpu, non_blocking=True)-0.4
                print(f'CIOSR_score: {out1}')
                print(f'model_score: {logits}')
                confidence1 = out1.data.max(1)[0]

                logits = torch.softmax(logits / 1, dim=1)

                confidence = logits.data.max(1)[0]



                for b in range(bsz):
                    probs[n] = confidence1[b]
                    open_labels[n] = 1
                    n += 1
                predictions = out1.data.max(1)[1]
                preval = logits.data.max(1)[0]
                preval1 = out1.data.max(1)[0]
                if (preval > 0.5 or preval1 > 0.5) and (predictions == labels.data):
                    correct += 1
                total += labels.size(0)
                #correct += (predictions == labels.data).sum()
                #pred_close.append((logits.data.cpu().numpy() + out1.data.cpu().numpy()) / 2)
                pred_close.append(out1.data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)

            oodlabel = torch.zeros_like(labels) - 1
            with torch.set_grad_enabled(False):
                gcfeatures = torch.load('resultsFU/PACS/feature_dict1_196.pt')

                # gcfeatures1 = torch.load('results/PACS/feature_dict2_183.pt')
                logits, cfeatures, t, map = net2(data)
                score, scores = predict_class(map, gcfeatures, 48)

                #  score2=predict_class2(cfeatures,gcfeatures1,128)

                cc = labels[0].item()

                if cc in score_dict:

                    a = score_dict[cc]
                    b = torch.tensor([score[cc]])
                    score_dict[cc] = torch.cat([a, b])


                else:
                    score_dict[cc] = torch.tensor([score[cc]])

                out1 = score.unsqueeze(0).cuda(args.gpu, non_blocking=True)-0.4

                confidence1 = out1.data.max(1)[0]

                logits = torch.softmax(logits / 1, dim=1)
                confidence = logits.data.max(1)[0]


                for b in range(bsz):
                    probs[n] = confidence1[b]
                    open_labels[n] = 0
                    n += 1
                pred_open.append(out1.data.cpu().numpy())
                labels_open.append(oodlabel.data.cpu().numpy())



    # Accuracy

    acc = (float(correct) * 100.) / float(total)
    #print('Acc: {:.5f}'.format(acc))
    pred_close = np.concatenate(pred_close, 0)
    pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open = np.concatenate(labels_open, 0)
    # F1 score Evaluation
    x1, x2 = np.max(pred_close, axis=1), np.max(pred_open, axis=1)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([x1, x2], axis=0)
    thr = 0.5
    open_pred = (total_pred > thr - 0.05).astype(np.float32)
    f = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')

    # AUROC score Evaluation
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    auc = roc_auc_score(open_labels, prob)

    return acc, auc, f


