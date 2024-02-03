import torch
import torch.nn as nn
import torch.nn.functional as F


class SNet(nn.Module):
    def __init__(self,input_size,input_features,r=16 ) -> None:
        super().__init__()
        #image_input -> H*W*C
        self.squeeze=nn.AvgPool2d(kernel_size=input_size) #1*1*C
        self.excite=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_features,input_features//r),
            nn.ReLU(),
            nn.Linear(input_features//r,input_features),
            nn.Sigmoid(),
        )
    
    
    def forward(self,x : torch.tensor):
        B,C,H,W=x.size()
        input=x
        x=self.squeeze(x)
        x=x.permute(0,2,3,1)
        x=self.excite(x)
        
        ## Shape -> B,H,W,C
        x=x.permute(0,3,1,2) #shape ->B,C,H,W
        
        return x*input
    
    
    
class ResSNet(nn.Module):
    def __init__(self, input_size,input_features) -> None:
        super().__init__()
        self.net=nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(),
            nn.Conv2d(input_features,input_features,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(input_features),
            nn.ReLU(),
            nn.Conv2d(input_features,input_features,kernel_size=3,padding=1,bias=False),
            SNet(input_size,input_features)
        )
    
    def forward(self,x:torch.tensor):
        residual=x
        x=self.net(x)
        return residual+x

####-----------------------------
#Testing
###----------------------------
if __name__=="__main__" :
    x=torch.randn(1,512,224,224,device='cuda')
    model=ResSNet(224,512).to('cuda')
    out=model(x)





        
