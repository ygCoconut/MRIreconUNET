import torch.nn as nn
import torch.nn.functional as F

s = 2 # "same" padding
class model1(nn.Module):

    def __init__(self):
        super(model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding = s)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5, padding = s)

        self.deconv2 = nn.ConvTranspose2d(16, 16, 4 ,stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(16, 6, 5, padding = s)
        self.conv4 = nn.Conv2d(6, 1, 1, padding = 0)
        self.conv5 = nn.Conv2d(1, 1, 1, padding = 0)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))

        x = F.relu(self.deconv2(x))
        x = F.relu(self.conv3(x))

        x = self.conv4(x)
        x = self.conv5(x)

        return x


class BNv0(nn.Module):

    def __init__(self):
        super(BNv0, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding = s)
        self.BN16 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5, padding = s)

        self.deconv2 = nn.ConvTranspose2d(16, 16, 4 ,stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(16, 6, 5, padding = s)
        self.BN6 = nn.BatchNorm2d(6)
        self.conv4 = nn.Conv2d(6, 1, 1, padding = 0)
        self.BN1 = nn.BatchNorm2d(1)
        self.conv5 = nn.Conv2d(1, 1, 1, padding = 0)


    def forward(self, x):
        x = self.pool(F.relu(self.BN16(self.conv1(x))))
        x = F.relu(self.BN16(self.conv2(x)))

        x = F.relu(self.deconv2(x))
        x = F.relu(self.BN6(self.conv3(x)))

        x = F.relu(self.BN1(self.conv4(x)))
        x = self.conv5(x)

        return x

class BNv1(nn.Module):

    def __init__(self):
        super(BNv1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding = s)
        self.BN16 = nn.BatchNorm2d(16)
        self.BN16n2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5, padding = s)

        self.deconv2 = nn.ConvTranspose2d(16, 16, 4 ,stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(16, 6, 5, padding = s)
        self.BN6 = nn.BatchNorm2d(6)
        self.conv4 = nn.Conv2d(6, 1, 1, padding = 0)
        self.BN1 = nn.BatchNorm2d(1)
        self.conv5 = nn.Conv2d(1, 1, 1, padding = 0)


    def forward(self, x):
        x = self.pool(F.relu(self.BN16(self.conv1(x))))
        x = F.relu(self.conv2(x)) #take BN out to avoid cycles

        x = self.deconv2(x)
        x = F.relu(self.BN6(self.conv3(x)))

        x = F.relu(self.BN1(self.conv4(x)))
        x = self.conv5(x)

        return x


class BN20channels(nn.Module):

    def __init__(self):
        super(BN20channels, self).__init__()
        self.conv1 = nn.Conv2d(20, 32, 5, padding = s)
        self.BN32 = nn.BatchNorm2d(32)
        self.BN64 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding = s)

        self.deconv2 = nn.ConvTranspose2d(64, 64, 4 ,stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 32, 5, padding = s)
        self.BN32_2 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 20, 1, padding = 0)
        self.BN20 = nn.BatchNorm2d(20)
        self.conv5 = nn.Conv2d(20, 20, 1, padding = 0)


    def forward(self, x):
        x = self.pool(F.relu(self.BN32(self.conv1(x))))
        x = F.relu(self.conv2(x)) #take BN out to avoid cycles

        x = self.deconv2(x)
        x = F.relu(self.BN32_2(self.conv3(x)))

        x = F.relu(self.BN20(self.conv4(x)))
        x = self.conv5(x)

        return x



'''
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))

        x = self.deconv2(x)
        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x
'''
