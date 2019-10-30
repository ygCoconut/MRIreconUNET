from models import UNet3d_parts
import torch.nn.functional as F
import torch.nn as nn

class UNet3d(nn.Module):
    # n_channels corresponds to the number of timeframes of the input
    # This value is equivalent for in and output, as 20 input timeframes
    # result in 20 output timeframes.
    def __init__(self, n_channels):
        super(UNet3d, self).__init__()
        self.doubleconv = UNet3d_parts.double_conv(n_channels, 64)
        self.down1 = UNet3d_parts.down(64, 128)
        self.down2 = UNet3d_parts.down(128, 256)

        self.up1 = UNet3d_parts.up(256, 128)
        self.up2 = UNet3d_parts.up(128, 64)
        self.endconv = nn.Conv3d(64, 1, 1)



# Add Concatenation !!!
    def forward(self, x):
        x = self.doubleconv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.endconv(x)

        return x
