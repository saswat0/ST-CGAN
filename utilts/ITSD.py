import math
import random
import numbers
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from PIL import Image
from torch import Tensor
import torchvision.transforms.functional as F

class Scale(object):
    # Scale the image during augmentation
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        output = []
        for img in imgs:
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                output.append(img)
                continue
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                output.append(img.resize((ow, oh), self.interpolation))
                continue
            else:
                oh = self.size
                ow = int(self.size * w / h)
            output.append(img.resize((ow, oh), self.interpolation))
        return output[0], output[1], output[2]

class ToTensor(object):
    def __call__(self, pic):
        return F.to_tensor(pic[0]), F.to_tensor(pic[1]), F.to_tensor(pic[2])

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor[0], self.mean, self.std, self.inplace), F.normalize(tensor[1], self.mean, self.std, self.inplace), F.normalize(tensor[2], self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
