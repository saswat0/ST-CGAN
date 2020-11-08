import os
import glob
import torch
import random

from PIL import Image
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import transforms

from ITSD import transforms

class ImageTransformOwn():
    # Transformation for own images
    def __init__(self, size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)
class ImageTransform():
    # Image Transform and Preprocessing
    def __init__(self, size=286, crop_size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = {'train': transforms.Compose([transforms.Scale(size=size),
                                                            transforms.RandomCrop(size=crop_size),
                                                            transforms.RandomHorizontalFlip(p=0.5),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)]),

                                'val': transforms.Compose([transforms.Scale(size=size),
                                                           transforms.RandomCrop(size=crop_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean, std)]),

                                'test': transforms.Compose([transforms.Scale(size=size),
                                                            transforms.RandomCrop(size=crop_size),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)])}

    def __call__(self, phase, img):
        return self.data_transform[phase](img)

class ImageDataset(data.Dataset):
    # Custom Dataset loader 
    def __init__(self, img_list, img_transform, phase):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self):
        return len(self.img_list['path_A'])

    def __getitem__(self, index):
        # Get tensor type of preprocessed image
        img = Image.open(self.img_list['path_A'][index]).convert('RGB')
        gt_shadow = Image.open(self.img_list['path_B'][index])
        gt = Image.open(self.img_list['path_C'][index]).convert('RGB')

        img, gt_shadow, gt = self.img_transform(self.phase, [img, gt_shadow, gt])

        return img, gt_shadow, gt
