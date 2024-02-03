import torch
import torch.nn as nn
import torch.nn.functional as f 
from typing import Tuple



class Encoder(nn.Module):
    def __init__(self,in_c,features) -> None:
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(in_c,features,kernel_size=3,padding=1),
                               nn.ReLU(),
                               nn.Conv2d(features,features,kernel_size=3,padding=1),
                               nn.ReLU(),
                              
                               
        )
        self.mp=nn.MaxPool2d((2,2))
    
    def forward(self,x):
        out=self.net(x)
        p=self.mp(out)
        return p,out
        

class Decoder(nn.Module):
    def __init__(self, in_c,features) -> None:
        super().__init__()
        self.conv1=nn.ConvTranspose2d(in_c,in_c//2,kernel_size=2,stride=2)
        self.net_decode=nn.Sequential(
            nn.Conv2d(features*2,features,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(features,features,kernel_size=3,padding=1),
            nn.ReLU()
            
        )
            
    
    def forward(self,x,skip) :
        out=self.conv1(x)

        out_concat=torch.cat([out,skip],dim=1)
        out_concat=self.net_decode(out_concat)
        
        return out_concat


class UNet(nn.Module):
    def __init__(self,C,n_classes=2 ) -> None:
        super().__init__()
        self.encode1=Encoder(C,64)
        self.encode2=Encoder(64,128)
        self.encode3=Encoder(128,256)
        self.encode4=Encoder(256,512)
        self.bottleneck=nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(1024,1024,kernel_size=3,padding=1),
            nn.ReLU()
        )
        
        self.decode1=Decoder(1024,512)
        self.decode2=Decoder(512,256)
        self.decode3=Decoder(256,128)
        self.decode4=Decoder(128,64)

        self.out=nn.Conv2d(64,n_classes,kernel_size=1)
    
    
    def forward(self,x:torch.tensor) :
        x1,skip1=self.encode1(x)
        x2,skip2=self.encode2(x1)
        x3,skip3=self.encode3(x2)
        x4,skip4=self.encode4(x3)
        
        xb=self.bottleneck(x4)
        
        out=self.decode1(xb,skip4)
        out=self.decode2(out,skip3)
        out=self.decode3(out,skip2)
        out=self.decode4(out,skip1)

        ###Segmentation map
        out=self.out(out)

        return out
        
        
#################
## Testing purpose

#######################
        
if __name__=="__main__" :
    x=torch.randn(4,1,576,576,device='cuda')
    model=UNet(1).to('cuda')
    out=model(x)
           
        