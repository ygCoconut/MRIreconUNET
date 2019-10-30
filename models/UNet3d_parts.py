
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.doubleconv = nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.doubleconv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.maxpool_doubleconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.maxpool_doubleconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  upsampling can be performed via interpolation if machine is not
        #  good enough for a backprop
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)

        self.doubleconv = double_conv(in_ch, out_ch)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = self.convdouble(x)
        return x


    # def forward(self, x1, x2):
    #     x1 = self.up(x1)
    #
    #     # input is CHW
    #     diffY = x2.size()[2] - x1.size()[2]
    #     diffX = x2.size()[3] - x1.size()[3]
    #
    #     x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
    #                     diffY // 2, diffY - diffY//2))
    #
    #     # for padding issues, see
    #     # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    #     # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    #
    #     x = torch.cat([x2, x1], dim=1)
    #     x = self.conv(x)
    #     return x
