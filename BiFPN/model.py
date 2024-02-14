import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange, repeat

from backbone import FakeBackBone



class DepthWiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=1,bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BiFPNBLock(nn.Module):
    def __init__(self, features, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        ## Conv 

        self.p7_to_p6 = DepthWiseConvBlock(features, features)
        self.p6_to_p5 = DepthWiseConvBlock(features, features)
        self.p5_to_p4 = DepthWiseConvBlock(features, features)
        self.p4_to_p3 = DepthWiseConvBlock(features, features)

        self.p3_to_p4 = DepthWiseConvBlock(features, features)
        self.p4_to_p5 = DepthWiseConvBlock(features, features)
        self.p5_to_p6 = DepthWiseConvBlock(features, features)
        self.p6_to_p7 = DepthWiseConvBlock(features, features)


        ## Features scaling layers
        self.p6_up_to_7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_up_to_6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_up_to_5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_up_to_4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_down_to_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p5_down_to_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p6_down_to_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p7_down_to_6 = nn.MaxPool2d(kernel_size=2, stride=2)


        ## Weight -- Learned coefficients (To adjust the importance of each feature map)
        self.p6_w = nn.Parameter(torch.ones(2,dtype=torch.float32,requires_grad=True))
        self.p6_w_relu=nn.ReLU()
        self.p5_w=nn.Parameter(torch.ones(2,dtype=torch.float32,requires_grad=True))
        self.p5_w_relu=nn.ReLU()
        self.p4_w=nn.Parameter(torch.ones(2,dtype=torch.float32,requires_grad=True))
        self.p4_w_relu=nn.ReLU()
        self.p3_w=nn.Parameter(torch.ones(2,dtype=torch.float32,requires_grad=True))
        self.p3_w_relu=nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()
    

    def forward(self, output_backbone : FakeBackBone ) :
        p3, p4, p5, p6, p7 = output_backbone

        # Weights for p6 and p7 to p6_first_node 
        p6_w=self.p6_w_relu(self.p6_w)
        weight= p6_w / (torch.sum(p6_w, dim=0) + self.epsilon)
        p6_down=self.p7_to_p6(weight[0] * p6  + weight[1] * self.p6_up_to_7(p7))

        # Weights for p5 and p6 to p5_first_node
        p5_w=self.p5_w_relu(self.p5_w)
        weight= p5_w / (torch.sum(p5_w, dim=0) + self.epsilon)
        p5_down=self.p6_to_p5(weight[0] * p5  + weight[1] * self.p5_up_to_6(p6_down))

        # Weights for p4 and p5 to p4_first_node
        p4_w=self.p4_w_relu(self.p4_w)
        weight= p4_w / (torch.sum(p4_w, dim=0) + self.epsilon)
        p4_down=self.p5_to_p4(weight[0] * p4  + weight[1] * self.p4_up_to_5(p5_down))

        # Weights for p3 and p4 to p3_second_node
        p3_w=self.p3_w_relu(self.p3_w)
        weight= p3_w / (torch.sum(p3_w, dim=0) + self.epsilon)
        p3_down=self.p4_to_p3(weight[0] * p3  + weight[1] * self.p3_up_to_4(p4_down))

        ## Down to up

        # Weights for p4, p4_first_node and p3_second_node to p4_second_node
        p4_w2=self.p4_w2_relu(self.p4_w2)
        weight= p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_up=self.p3_to_p4(weight[0] * p4  + weight[1] * p4_down + weight[2] * self.p4_down_to_3(p3_down))

        # Weights for p5, p5_first_node and p4_second_node to p5_second_node
        p5_w2=self.p5_w2_relu(self.p5_w2)
        weight= p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_up=self.p4_to_p5(weight[0] * p5  + weight[1] * p5_down + weight[2] * self.p5_down_to_4(p4_up))

        # Weights for p6, p6_first_node and p5_second_node to p6_second_node
        p6_w2=self.p6_w2_relu(self.p6_w2)
        weight= p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_up=self.p5_to_p6(weight[0] * p6  + weight[1] * p6_down + weight[2] * self.p6_down_to_5(p5_up))

        # Weights for p7, p6_second_node to p7_second_node
        p7_w2=self.p7_w2_relu(self.p7_w2)
        weight= p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_up=self.p6_to_p7(weight[0] * p7  + weight[1] * self.p7_down_to_6(p6_up))


        return p3_down,p4_up,p5_up,p6_up,p7_up




class BiFPN(nn.Module):
    def __init__(self,features, n_blocks=4) -> None:
        super().__init__()
        self.n_bifpn=nn.Sequential(*[BiFPNBLock(features) for _ in range(n_blocks)])
    

    def forward(self, output_backbone : FakeBackBone ) :
        p3, p4, p5, p6, p7 = output_backbone
        p3_out, p4_out, p5_out, p6_out, p7_out= self.n_bifpn(output_backbone)
        return p3_out, p4_out, p5_out, p6_out, p7_out




## Test
    
if __name__=='__main__':
    inputs_backbone=(torch.rand(1,256,32,32),torch.rand(1,512,16,16),torch.rand(1,1024,8,8))
    model=BiFPN(64)
    out=model(FakeBackBone([256,512,1024],64)(inputs_backbone))









        


