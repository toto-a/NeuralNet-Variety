import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class FakeBackBone(nn.Module):
    def __init__(self,size :list[int], feature_size ) -> None:
        super().__init__()
        self.p3= nn.Conv2d(size[0],feature_size,1,1,0)
        self.p4=nn.Conv2d(size[1],feature_size,1,1,0)
        self.p5=nn.Conv2d(size[2],feature_size,1,1,0)

        self.p6=nn.Conv2d(size[2],feature_size,3,2,1)
        self.p7=ConvBlock(feature_size,feature_size,3,2,1)

    
    def forward(self,inputs: Tuple[torch.tensor]) :
        c3,c4,c5=inputs

        p3=self.p3(c3)
        p4=self.p4(c4)
        p5=self.p5(c5)
        p6=self.p6(c5)
        p7=self.p7(p6)

        return [p3,p4,p5,p6,p7]

