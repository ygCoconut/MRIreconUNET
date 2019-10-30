import torch.nn as nn
import torch.nn.functional as F

s = 2 # "same" padding
class Net1(nn.Module):
    
    #F.conv2d(input, self.weight, self.bias, self.stride,
    
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding = s)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = s)
        
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

def foo(hi): #self ??
    print("hello foo")
    return hi

class Net2(nn.Module):
    
    #F.conv2d(input, self.weight, self.bias, self.stride,
    
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding = s)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = s)
        
        self.conv3 = nn.Conv2d(16, 6, 5, padding = s) 
        self.conv4 = nn.Conv2d(6, 1, 1, padding = 0)
        self.conv5 = nn.Conv2d(1, 1, 1, padding = 0)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.conv5(x)

        return x