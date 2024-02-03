import torch
import torch.nn as nn
import torch.nn.functional as F
from snet import SNet

class Spatial_Attention (nn.Module):
    def __init__(self,kernel_size=7 ) -> None:
        super().__init__()
        self.spa=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2,dilation=1,bias=False) ,##padding for the same size
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x_avg=x.mean(dim=1,keepdim=True)
        x_max=x.max(dim=1,keepdim=True)[0]##torch.max return the corresponding indices
        x_conv=self.spa(torch.cat((x_avg,x_max),dim=1))
        
        return x_conv*x



class Channel_Attention(nn.Module) :
    def __init__(self, in_features,r=16) -> None:
        super().__init__()
        self.ca=nn.Sequential(
            nn.Linear(in_features,in_features//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_features//r,in_features)
            
        )

        self.out_ca=nn.Sigmoid()
    
    def forward(self,x):
        x_max=F.adaptive_max_pool2d(x,(1,1))
        x_avg=F.adaptive_avg_pool2d(x,(1,1))
        x_shared=self.ca(x_max.permute(0,2,3,1))+self.ca(x_avg.permute(0,2,3,1))
        x_shared=x_shared.permute(0,3,1,2)
        
        return self.out_ca(x_shared)*x



class CBAM(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.CA=Channel_Attention(in_features)
        self.SA=Spatial_Attention(in_features)
        
    def forward(self,x) :
        out=x
        x=self.CA(x)
        x=self.SA(x)
        
        return out+x
    



if __name__=='__main__':
    x=torch.randn(1,512,224,224,device='cuda')
    model=CBAM(512).to('cuda')
    out=model(x)



    
        
        

        

