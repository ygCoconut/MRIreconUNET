#Assembly of the network
# Code from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

from models import UNet2d_parts
import torch.nn.functional as F
import torch.nn as nn


class UNet2D(nn.Module):
    # n_channels corresponds to the number of timeframes of the input
    # This value is equivalent for in and output, as 20 input timeframes
    # result in 20 output timeframes.
	def __init__(self):
		super(UNet2D, self).__init__()
		self.doubleconvstart = UNet2d_parts.double_convo(20, 64)
		self.down1 = UNet2d_parts.down_sampling(64, 128)
		self.down2 = UNet2d_parts.down_sampling(128, 256)
		self.down3 = UNet2d_parts.down_sampling(256, 512)
		self.down4 = UNet2d_parts.down_sampling(512, 512)
		self.up1 = UNet2d_parts.up_sampling(1024, 256)
		self.up2 = UNet2d_parts.up_sampling(512, 128)
		self.up3 = UNet2d_parts.up_sampling(256, 64)
		self.up4 = UNet2d_parts.up_sampling(128, 64)
		self.doubleconvfinal = UNet2d_parts.double_convo(64, 20)
		self.onexoneconv = nn.Conv2d(20, 20, 1)
# the final convolution is executed without nn.Sequential for better
# visualization with tensorboard graph
		# self.final = nn.Sequential(
        #     UNet2d_parts.double_convo(64, n_channels),
        #     nn.Conv2d(n_channels, n_channels, 1)
        #     )


	def forward(self, x):
		x1 = self.doubleconvstart(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.doubleconvfinal(x)
		x = self.onexoneconv(x)
		return x
