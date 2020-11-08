import torch
import torch.nn as nn
import torch.nn.functional as F

from net.CV import CvTi, Cvi

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(Generator, self).__init__()

        self.Cv0 = Cvi(input_channels, 64)
        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')
        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')
        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')
        self.Cv4 = Cvi(512, 512, before='LReLU', after='BN')
        self.Cv5 = Cvi(512, 512, before='LReLU')

        self.CvT6 = CvTi(512, 512, before='ReLU', after='BN')
        self.CvT7 = CvTi(1024, 512, before='ReLU', after='BN')
        self.CvT8 = CvTi(1024, 256, before='ReLU', after='BN')
        self.CvT9 = CvTi(512, 128, before='ReLU', after='BN')
        self.CvT10 = CvTi(256, 64, before='ReLU', after='BN')
        self.CvT11 = CvTi(128, output_channels, before='ReLU', after='Tanh')

    def forward(self, input):
        # Encoder Network
        out_0 = self.Cv0(input)
        out_1 = self.Cv1(out_0)
        out_2 = self.Cv2(out_1)
        out_3 = self.Cv3(out_2)
        out_4_1 = self.Cv4(out_3)
        out_4_2 = self.Cv4(out_4_1)
        out_4_3 = self.Cv4(out_4_2)
        out_5 = self.Cv5(out_4_3)

        # Decoder Network
        out_6 = self.CvT6(out_5)

        cat1_1 = torch.cat([out_6, out_4_3], dim=1)
        out_7_1 = self.CvT7(cat1_1)
        cat1_2 = torch.cat([out_7_1, out_4_2], dim=1)
        out_7_2 = self.CvT7(cat1_2)
        cat1_3 = torch.cat([out_7_2, out_4_1], dim=1)
        out_7_3 = self.CvT7(cat1_3)

        cat2 = torch.cat([out_7_3, out_3], dim=1)
        out_8 = self.CvT8(cat2)

        cat3 = torch.cat([out_8, out_2], dim=1)
        out_9 = self.CvT9(cat3)

        cat4 = torch.cat([out_9, out_1], dim=1)
        out_10 = self.CvT10(cat4)

        cat5 = torch.cat([out_10, out_0], dim=1)
        out = self.CvT11(cat5)

        return out
