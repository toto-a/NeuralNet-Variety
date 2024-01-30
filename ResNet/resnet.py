import torch
import torch.nn as nn
import torch.nn.functional as f 


class Block(nn.Module):
    def __init__(self,in_chan,out_chan,stride,downsample=None) -> None:
        super().__init__()
        self.b1=self._block(in_chan,out_chan,kernel_size=3,padding=1,stride=stride)
        self.b2=self.block(out_chan,out_chan,kernel_size=3,padding=1,stride=1)
        self.relu=nn.ReLU()
        self.down=downsample
    
    def _block(self,in_chan,out_chan,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
    
    
    def forward(self,x):
        _residual=x
        x=self.b1(x)
        x=self.b2(x)
        
        if self.down is not None :
            x=self.down(x)
        x=x+_residual
        out=self.relu(x)

        return out
        
        