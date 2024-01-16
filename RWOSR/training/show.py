import time
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
from loss_reweighting1 import predict_class,predict_class1,predict_class2
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def validate(val_loader, model, test=True):
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
    t1 = {}
    t1.clear()
    nn=0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image_path = 'data/train/bird1/19.jpg'
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor1 = preprocess(image)

    input_tensor=input_tensor1.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    print(f'input_tensor: {input_tensor.shape}')

    importance_map = torch.zeros_like(input_tensor1)

    # 进行预测
    with torch.no_grad():
        predictions, flatten_features,t,map = model(input_tensor)

    # 获取预测结果中的最高概率类别
    target_class = torch.argmax(predictions, dim=1)

    # 对每个像素点进行计算
    for h in range(importance_map.shape[1]):
        for w in range(importance_map.shape[2]):
            # 像素点置零
            modified_input = input_tensor1.clone()
            modified_input[:, h, w] = 0.0

            # 重新预测
            with torch.no_grad():
                modified_predictions,flatten_features,t,map = model(modified_input.unsqueeze(0).to(device))

            # 计算贡献程度
            difference = modified_predictions[0, target_class] - predictions[0, target_class]
            importance_map[:, h, w] = difference

    # 可视化重要度图
    heatmap = torch.abs(importance_map).max(dim=0)[0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化处理
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(1425,1900), mode='bilinear',
                            align_corners=False)  # 插值操作
    heatmap = heatmap.squeeze()  # 去除额外的维度
    heatmap = heatmap.cpu().numpy()


    heatmap_image = Image.fromarray((heatmap * 255).astype('uint8'))
    heatmap_image.save('heatmap3.jpg')
    # 可以使用matplotlib库来显示热图






    # with torch.no_grad():
    #     feature_map = model.module.conv1(input_tensor)
    #     feature_map = model.module.bn1(feature_map)
    #     feature_map = model.module.relu(feature_map)
    #     feature_map = model.module.maxpool(feature_map)
    #     feature_map = model.module.layer1(feature_map)
    #     feature_map = model.module.layer2(feature_map)
    #     feature_map = model.module.layer3(feature_map)
    #     feature_map = model.module.layer4(feature_map)
    #
    # upsampled_map = F.interpolate(feature_map, size=image.size, mode='bilinear', align_corners=False)
    #
    #
    #         # 将特征图映射到原图上
    # heatmap = torch.mean(upsampled_map, dim=1).squeeze().cpu().numpy()
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = np.transpose(heatmap, (1, 0))
    # heatmap_image = Image.fromarray((heatmap * 255).astype('uint8'))
    # heatmap_image.save('heatmap1.jpg')
    # alpha = 0.5
    #
    # # 将热力图与原始图像进行混合
    # blended_image = Image.blend(image, heatmap_image, alpha)
    #
    # # 保存混合图像
    # blended_image.save('blended_image.jpg')
    #
    # print(f'heatmap: {heatmap.shape}')
    #         # 叠加热力图到原图
    # save_path = 'heatmap.jpg'
    # plt.imshow(image)
    # plt.imshow(heatmap, alpha=0.5, cmap='jet')
    # plt.axis('off')
    # plt.savefig(save_path, format='jpg')
    # plt.show()
    # return 2

