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

    def __call__(self, mode, img):
        return self.data_transform[mode](img)

class ImageDataset(data.Dataset):
    # Custom Dataset loader 
    def __init__(self, img_list, img_transform, mode):
        self.img_list = img_list
        self.img_transform = img_transform
        self.mode = mode

    def __len__(self):
        return len(self.img_list['path_A'])

    def __getitem__(self, index):
        # Get tensor type of preprocessed image
        img = Image.open(self.img_list['path_A'][index]).convert('RGB')
        gt_shadow = Image.open(self.img_list['path_B'][index])
        gt = Image.open(self.img_list['path_C'][index]).convert('RGB')

        img, gt_shadow, gt = self.img_transform(self.mode, [img, gt_shadow, gt])

        return img, gt_shadow, gt

def create_path(mode="train", rate=0.8):
    # Generate filepath list for train, validation and test images
    
    random.seed(44)

    rootpath = './dataset/' + mode + '/'
    files_name = os.listdir(rootpath + mode + '_A')

    if mode=='train':
        random.shuffle(files_name)
    elif mode=='test':
        files_name.sort()

    path_A = []
    path_B = []
    path_C = []

    for name in files_name:
        path_A.append(rootpath + mode + '_A/'+name)
        path_B.append(rootpath + mode + '_B/'+name)
        path_C.append(rootpath + mode + '_C/'+name)

    num = len(path_A)

    if mode=='train':
        path_A, path_A_val = path_A[:int(num*rate)], path_A[int(num*rate):]
        path_B, path_B_val = path_B[:int(num*rate)], path_B[int(num*rate):]
        path_C, path_C_val = path_C[:int(num*rate)], path_C[int(num*rate):]
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
        path_list_val = {'path_A': path_A_val, 'path_B': path_B_val, 'path_C': path_C_val}
        return path_list, path_list_val

    elif mode=='test':
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
        return path_list
