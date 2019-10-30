

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_convo(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_convo, self).__init__()
        self.doubleconv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.doubleconv(x)
        return x

class down_sampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_sampling, self).__init__()
        self.maxpoolconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_convo(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.maxpoolconv(x)
        return x



class up_sampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_sampling, self).__init__()
        self.upconvolution = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.doubleconv = double_convo(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upconvolution(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.doubleconv(x)
        return x


class final_conv(nn.Module):
    def __init__(self, num_ch):
        super(final_conv, self).__init__()
        self.finalconv = nn.Conv2d(num_ch, num_ch, 1)

    def forward(self, x):
        x = self.finalconv(x)
        return x
