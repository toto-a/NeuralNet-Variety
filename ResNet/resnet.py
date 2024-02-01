import torch
import torch.nn as nn
import torch.nn.functional as f 
from typing import Tuple


class Block(nn.Module):
    def __init__(self,in_chan,out_chan,stride=1,downsample=None) -> None:
        super().__init__()
        self.b1=self._block(in_chan,out_chan,kernel_size=3,stride=stride,padding=1)
        self.b2=self._block(out_chan,out_chan,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        self.down=downsample
    
    def _block(self,in_chan,out_chan,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
    
    
    def forward(self,x):
        _residual=x
        out=self.b1(x)
        out=self.b2(out)
        
        if self.down is not None :
            _residual=self.down(x)
        out=out+_residual
        out=self.relu(out)

        return out
        
        
class ResNet(nn.Module):
    def __init__(self, block : list(), num_classes : int =10):
        super().__init__()
        self.inplanes=64
        self.conv1=nn.Sequential(nn.Conv2d(3,64,kernel_size=7,stride=2, padding=3),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                )
        self.mp=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.block1=self._make_block(64,1,block[0])
        self.block2=self._make_block(128,stride=2,block=block[1])
        self.block3=self._make_block(256,stride=2,block=block[2])
        self.block4=self._make_block(512,stride=2,block=block[3])
        
        self.avg_pool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.classifier=nn.Linear(512,num_classes)
        
    def _make_block(self,out_c,stride,block
                    ):
         
        downsample=None
        if stride!=1 or self.inplanes!=out_c :
             
            downsample=nn.Sequential(
                    nn.Conv2d(self.inplanes,out_c,kernel_size=1,stride=stride),
                    nn.BatchNorm2d(out_c)
                )

        _nn=[Block(self.inplanes,out_c,stride,downsample)]
        self.inplanes=out_c
        for _ in range(block-1) :
            _nn.append(Block(self.inplanes,out_c))
        
        
        return nn.Sequential(*_nn)
        

        
    def forward(self,x:torch.tensor):
        out=self.conv1(x)
        out=self.mp(out)
        
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.block4(out)
        
        out=self.avg_pool(out)
        out=out.view(x.size(0),-1)

        out=self.classifier(out)
        return out

         