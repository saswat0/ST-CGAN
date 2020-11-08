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

class Discriminator(nn.Module):
    def __init__(self, input_channels=4):
        super(Discriminator, self).__init__()

        self.Cv0 = Cvi(input_channels, 64)
        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')
        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')
        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')
        self.Cv4 = Cvi(512, 1, before='LReLU', after='sigmoid')

    def forward(self, input):
        out_0 = self.Cv0(input)
        out_1 = self.Cv1(out_0)
        out_2 = self.Cv2(out_1)
        out_3 = self.Cv3(out_2)
        out = self.Cv4(out_3)

        return out

# Testing Codes
if __name__ == "__main__":
    # BCHW
    input_size = (3, 3, 256, 256)
    input = torch.ones(input_size)
    l1 = nn.L1Loss()
    input.requires_grad = True

    # Conv test
    conv = Cvi(3, 3)
    conv2 = Cvi(3, 3, before='ReLU', after='BN')
    output = conv(input)
    output2 = conv2(output)
    # print(output.shape)
    # print(output2.shape)
    loss = l1(output, torch.randn(3, 3, 128, 128))
    loss.backward()
    # print(loss.item())

    convT = CvTi(3, 3)
    outputT = convT(output)
    # print(outputT.shape)

    #Generator test
    model = Generator()
    output = model(input)
    # print(output.shape)
    loss = l1(output, torch.randn(3, 1, 256, 256))
    loss.backward()
    # print(loss.item())

    #Discriminator test
    input_size = (3, 4, 256, 256)
    input = torch.ones(input_size)
    l1 = nn.L1Loss()
    input.requires_grad = True
    model = Discriminator()
    output = model(input)
    # print(output.shape)
