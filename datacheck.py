import os
from PIL import Image
import PIL
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
traindir = "data/train"
test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])),
        batch_size=1)

for i, (images, targets) in enumerate(test_loader):
    print(f'Tensor: {targets}')






