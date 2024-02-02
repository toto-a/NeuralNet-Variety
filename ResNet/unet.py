import torch
import torch.nn as nn
import torch.nn.functional as f 
from typing import Tuple



class Encoder(nn.Module):
    def __init__(self,in_c,features) -> None:
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(in_c,features,kernel_size=3),
                               nn.ReLU(),
                               nn.Conv2d(features,features,kernel_size=3),
                               nn.ReLU(),
                               nn.MaxPool2d(kernel_size=2,stride=2)
                               
        )
    
    def forward(self,x):
        return self.net(x)
        
