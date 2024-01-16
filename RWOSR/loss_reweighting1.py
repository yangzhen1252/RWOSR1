# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sd(x):
    return np.std(x, axis=0, ddof=1)


def sd_gpu(x):
    return torch.std(x, dim=0)


def normalize_gpu(x):
    x = F.normalize(x, p=1, dim=1)
    return x


def normalize(x):
    mean = np.mean(x, axis=0)
    std = sd(x)
    std[std == 0] = 1
    x = (x - mean) / std
    return x
def lossb_expect1(feature_dict, num_f, sum=True):
    n = len(feature_dict)
    feature_tensors = [torch.tensor(f) for f in feature_dict]
    f_vectors = random_fourier_features_gpu(feature_tensors, num_f=num_f, sum=sum)
    loss = 0

    for i, features in enumerate(f_vectors):
        fi = features[i]

        cov1 = cov(fi)  # 计算fi的协方差

        cov_matrix = cov1 * cov1  # fi和fi的协方差是自相关
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss

def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z

# def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
#     if num_f is None:
#         num_f = 6
#     n = x.size(0)
#     r = x.size(1)
#     x = x.view(n, r, 1)
#     c = x.size(2)
#     if sigma is None or sigma == 0:
#         sigma = 1
#     if w is None:
#         w = 1 / sigma * (torch.randn(size=(num_f, c)))
#         b = 2 * np.pi * torch.rand(size=(r, num_f))
#         b = b.repeat((n, 1, 1))
#
#     Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())
#
#     mid = torch.matmul(x.cuda(), w.t().cuda())
#
#     mid = mid + b.cuda()
#     mid -= mid.min(dim=1, keepdim=True)[0]
#     mid /= mid.max(dim=1, keepdim=True)[0].cuda()
#     mid *= np.pi / 2.0
#
#     if sum:
#         Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
#     else:
#         Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)
#
#     return Z

def random_fourier_features_gpu1(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    x=x.squeeze()
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z


def predict_class(sample_feature, feature_dict, num_f, sum=True):
    # print(f'sample_feature: {sample_feature.shape}')

    f_vectors_dict = {}  # 用于保存每个类别的随机傅里叶特征
    sample_feature = sample_feature.squeeze()
    sample_feature=sample_feature.view(sample_feature.size(0), -1)

    # 计算样本特征的随机傅里叶特征
    sample_f_vectors = random_fourier_features_gpu(sample_feature, num_f=num_f, sum=sum)


    # 计算特征字典中各类别特征的随机傅里叶特征
    for category, features in feature_dict.items():
        f_vectors1 = random_fourier_features_gpu(features, num_f=num_f, sum=sum)

        f_vectors_dict[category] = f_vectors1

    scores = {}  # 保存类别概率分数
    scores1 = {}
    # print(f'sample_f_vectors1: {sample_f_vectors.shape}')
    # 计算样本特征与特征字典中各类别特征的相关性，得到概率分数
    for category, f_vectors in f_vectors_dict.items():
        # sample_f_vectors1 = sample_f_vectors.squeeze().permute(1, 0)  # 去除不必要的维度
        #
        # print(f'sample_f_vectors: {sample_f_vectors.shape}')
        # print(f'_vectors: {f_vectors.shape}')
        # f_vectors = f_vectors.squeeze().permute(1, 0)
        sample_f_vectors1 = sample_f_vectors.view(sample_f_vectors.size(0), -1) # 去除不必要的维度
        f_vectors = f_vectors.view(f_vectors.size(0), -1)
        # print(f'sample_f_vectors1: { sample_f_vectors1.shape}')
        # print(f'f_vectors: {f_vectors.shape}')
        cov1=covn(sample_f_vectors1,f_vectors)

        mse = torch.mean((sample_f_vectors1- f_vectors) ** 2)
        cov_matrix = cov1*cov1# 计算协方差矩阵

        score = torch.sum(cov_matrix) - torch.trace(cov_matrix)  # 计算概率分数
        scores[category] = score.item()
    # for key, tensor in scores.items():
    #     print(f'Tensor: {key}')
    #     print(f'Shape: {tensor}')
    for category in range(len(scores)):
        scores1[category] =scores[category]
    # for key, tensor in scores1.items():
    #         print(f'Tensor1: {key}')
    #         print(f'Shape1: {tensor}')
    #scores_tensor = torch.cat([torch.Tensor([scoren]) for scoren in scores.values()])
    scores_tensor = torch.cat([torch.Tensor([n]) for n in scores1.values()])  # 将概率分数张量拼接为一个张量
   # scores_tensor = torch.clamp(scores_tensor, min=0)  # 将scores_tensor中的负值设置为零
    #print(f'scores_tensor: {scores_tensor.shape}')
    return scores_tensor*100,scores


def predict_class2(sample_feature, feature_dict, num_f, sum=True):
    f_vectors_dict = {}  # 用于保存每个类别的随机傅里叶特征
   # sample_feature = sample_feature.unsqueeze()
    sample_feature=sample_feature

    # 计算样本特征的随机傅里叶特征
    sample_f_vectors = random_fourier_features_gpu(sample_feature, num_f=num_f, sum=sum)


    # 计算特征字典中各类别特征的随机傅里叶特征
    for category, features in feature_dict.items():
        f_vectors1 = random_fourier_features_gpu(features.unsqueeze(0), num_f=num_f, sum=sum)

        f_vectors_dict[category] = f_vectors1

    scores = {}  # 保存类别概率分数
    scores1 = {}
    # print(f'sample_f_vectors1: {sample_f_vectors.shape}')
    # 计算样本特征与特征字典中各类别特征的相关性，得到概率分数
    for category, f_vectors in f_vectors_dict.items():
        # sample_f_vectors1 = sample_f_vectors.squeeze().permute(1, 0)  # 去除不必要的维度
        #
        # f_vectors = f_vectors.squeeze().permute(1, 0)
        sample_f_vectors1 = sample_f_vectors # 去除不必要的维度
        # print(f'sample_f_vectors1: { sample_f_vectors1.shape}')
        # print(f'f_vectors: {f_vectors.shape}')
        cov1=covn(sample_f_vectors1.squeeze(),f_vectors.squeeze())
      #  print(f'cov1: {cov1.shape}')

        mse = torch.mean((sample_f_vectors1- f_vectors) ** 2)
        cov_matrix = cov1*cov1# 计算协方差矩阵

        score = torch.sum(cov_matrix) - torch.trace(cov_matrix)  # 计算概率分数
        scores[category] = score.item()
    # for key, tensor in scores.items():
    #     print(f'Tensor: {key}')
    #     print(f'Shape: {tensor}')
    for category in range(len(scores)):
        scores1[category] =scores[category]
    # for key, tensor in scores1.items():
    #         print(f'Tensor1: {key}')
    #         print(f'Shape1: {tensor}')
    #scores_tensor = torch.cat([torch.Tensor([scoren]) for scoren in scores.values()])
    scores_tensor = torch.cat([torch.Tensor([score]) for score in scores1.values()])  # 将概率分数张量拼接为一个张量
   # scores_tensor = torch.clamp(scores_tensor, min=0)  # 将scores_tensor中的负值设置为零
    #print(f'scores_tensor: {scores_tensor.shape}')
    return scores_tensor



def predict_class1(sample_feature, num_f, sum=True):


    # 计算样本特征的随机傅里叶特征
    sample_f_vectors = random_fourier_features_gpu1(sample_feature, num_f=num_f, sum=sum)


    sample_f_vectors1 = sample_f_vectors.squeeze().permute(1, 0)  # 去除不必要的维度




    cov1=cov(sample_f_vectors1)


    cov_matrix = cov1*cov1# 计算协方差矩阵

    score = torch.sum(cov_matrix) - torch.trace(cov_matrix)  # 计算概率分数

    return score


def lossb_expect(feature_dict, num_f, sum=True):
    f_vectors_dict = {}  # 用于保存每个类别的随机傅里叶特征
    loss = Variable(torch.FloatTensor([0]).cuda())
    for category, x in feature_dict.items():
        #print(f'x: {x}')
        f_vectors = random_fourier_features_gpu(x, num_f=num_f, sum=sum)
        # print(f'x: { f_vectors.shape}')
        f_vectors_dict[category] = f_vectors
    # for key, tensor in f_vectors_dict.items():
    #         print(f'Tensor: {key}')
    #         print(f'Shape: {tensor}')
        #     print()  # 打印空行以区分不同张量

    for i in range(len(f_vectors_dict)):
        for j in range(i + 1, len(f_vectors_dict)):
            # print(f'f_vectors_dict[i]: {f_vectors_dict[i]}')
            # print(f'f_vectors_dict[j]: {f_vectors_dict[j]}')
            fi = f_vectors_dict[i].view(f_vectors_dict[i].size(0), -1)# 去除不必要的维度
            fj = f_vectors_dict[j].view(f_vectors_dict[j].size(0), -1)
            # print(f'fi: {fi}')
            # print(f'fj: {fj}')
            cov1  = covn(fi,fj)  # 计算 fi 和 fj 之间的协方差
            # print(f'cov1: {cov1}')
             # 计算 fj 和 fi 之间的协方差
            cov_matrix = cov1 * cov1

            loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
            # print(f'a: {torch.sum(cov_matrix) }')
            # print(f'b: {torch.trace(cov_matrix)}')
    return loss



def lossb_expecttt(a,b, num_f, sum=True):

    loss = Variable(torch.FloatTensor([0]).cuda())
    a = random_fourier_features_gpu(a, num_f=num_f, sum=sum)
    b = random_fourier_features_gpu(b, num_f=num_f, sum=sum)


    fi = a.view(a.size(0), -1)# 去除不必要的维度
    fj = b.view(b.size(0), -1)
    # print(f'fi: {fi.shape}')
    # print(f'fj: {fj.shape}')
    cov1  = covn(fi,fj)  # 计算 fi 和 fj 之间的协方差
    # print(f'cov1: {cov1.shape}')
             # 计算 fj 和 fi 之间的协方差
    cov_matrix = cov1 * cov1

    loss = torch.sum(cov_matrix) - torch.trace(cov_matrix)
            # print(f'a: {torch.sum(cov_matrix) }')
            # print(f'b: {torch.trace(cov_matrix)}')
    # print(f'loss: {loss}')
    return 1-loss








def lossb_expectn(feature_dict):

    loss = Variable(torch.FloatTensor([0]).cuda())


    for i in range(len(feature_dict)):
        for j in range(i + 1, len(feature_dict)):

            fi = feature_dict[i].unsqueeze(0) # 去除不必要的维度
            fj = feature_dict[j].unsqueeze(0)

            cov1  = covn(fi, fj)  # 计算 fi 和 fj 之间的协方差
            # 计算 fj 和 fi 之间的协方差

            cov_matrix = cov1 * cov1

            loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss

def lossb_expectt(feature_dict,output, num_f, sum=True):

    loss = Variable(torch.FloatTensor([0]).cuda())
    output=output


    for i in range(len(feature_dict)):

        fi = feature_dict[i].view(6, -1) # 去除不必要的维度
        fj = output.view(6, -1)

        cov1  = cov1(fi)  # 计算 fi 和 fj 之间的协方差
        cov2 = cov1(fj)  # 计算 fj 和 fi 之间的协方差

        cov_matrix = cov1 * cov2

        score= torch.sum(cov_matrix) - torch.trace(cov_matrix)


    return loss/10000000

def covn(x, y, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), y) / n
        ex = torch.mean(x, dim=0).view(-1, 1)
        ey = torch.mean(y, dim=0).view(-1, 1)
        res = cov - torch.matmul(ex, ey.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), y)
        ex = torch.sum(w * x, dim=0).view(-1, 1)
        ey = torch.sum(w * y, dim=0).view(-1, 1)
        res = cov - torch.matmul(ex, ey.t())

    return res

# def lossb_expect(feature_dict, num_f, sum=True):
#     x = torch.stack(list(feature_dict.values()))  # 将特征字典的值转换为张量并堆叠
#     f_vectors = random_fourier_features_gpu(x, num_f=num_f, sum=sum)
#     loss = torch.tensor(0, dtype=torch.float32, device='cuda')
#
#     keys = list(feature_dict.keys())  # 获取特征字典的键（特征名称）
#
#     for i in range(len(keys)):
#         for j in range(i + 1, len(keys)):
#             fi = f_vectors[i].squeeze()  # 去除不必要的维度
#             fj = f_vectors[j].squeeze()
#
#             cov1  = cov(fi)  # 计算 fi 和 fj 之间的协方差
#             cov2 = cov(fj)  # 计算 fj 和 fi 之间的协方差
#
#             cov_matrix = cov1 * cov2
#             print(f'a1: {cov1.shape}')
#             print(f'b1: {cov1.shape}')
#             loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
#             print(f'a: {torch.sum(cov_matrix)}')
#             print(f'b: {torch.trace(cov_matrix)}')
#
#     return loss/(1000000000*4)

# def lossb_expect(feature_dict,  num_f, sum=True):
#     n = len(feature_dict)
#     feature_tensors = [torch.tensor(f) for f in feature_dict]
#     f_vectors = random_fourier_features_gpu(feature_tensors, num_f=num_f, sum=sum)
#     loss = Variable(torch.FloatTensor([0]).cuda())
#
#
#     for i in range(n):
#         for j in range(i + 1, n):
#             fi = f_vectors[i]
#             fj = f_vectors[j]
#
#             cov1 = cov(fi)  # Calculate covariance between fi and fj
#             cov2 = cov(fj)  # Calculate covariance between fj and fi
#
#             cov_matrix = cov1 * cov2
#             loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
#
#     return loss


# def lossb_expect1(feature_dict, num_f, sum=True):
#     num_classes = len(feature_dict)
#     feature_tensors = [feature_dict[i] for i in range(num_classes)]  # 将特征字典转换为列表
#     f_vectors = random_fourier_features_gpu(feature_tensors, num_f=num_f, sum=sum)
#
#     loss = torch.tensor(0, dtype=torch.float32, device='cuda')
#     for i in range(num_classes):
#         fi = f_vectors[i].squeeze()  # 去除不必要的维度
#
#         cov1 = cov(fi)  # 计算特征向量的协方差矩阵
#         cov2 = cov(fi, w=None)  # 计算特征向量的自协方差矩阵
#
#         loss += torch.sum(cov1) - torch.trace(cov1) + torch.sum(cov2) - torch.trace(cov2)
#
#     return loss

def lossc(inputs, target, weight):
    loss = nn.NLLLoss(reduce=False)
    return loss(inputs, target).view(1, -1).mm(weight).view(1)


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res



def lossq(cfeatures, cfs):
    return - cfeatures.pow(2).sum(1).mean(0).view(1) / cfs


def lossn(cfeatures):
    return cfeatures.mean(0).pow(2).mean(0).view(1)






if __name__ == '__main__':
    pass
