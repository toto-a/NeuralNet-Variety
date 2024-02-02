import torch
import torch.nn as nn
import torch.nn.functional as F



class VGG(nn.Module):
    def __init__(self, f_in,f_out,n_classes=10) -> None:
        super().__init__()
        self.b1=self._block(f_in,f_out)
        self.b2=self._block(f_out,f_out*2)
        self.b3=self._block(f_out*2,f_out*4,next=True)
        self.b4=self._block(f_out*4,f_out*8,next=True)
        self.b5=self._block(f_out*8,f_out*8,next=True)

        self.l1=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
        )
        self.l2=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(4096,4096),
            nn.ReLU(),
        )

        self.classifier=nn.Linear(4096,n_classes)

        
       
    
    def _block(self,features_in,features_out,next=False) :

            block=nn.Sequential(
            nn.Conv2d(features_in,features_out,kernel_size=3,padding=1),
            nn.BatchNorm2d(features_out),
            nn.ReLU(),
            nn.Conv2d(features_out,features_out,kernel_size=3,padding=1),
            nn.BatchNorm2d(features_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
                )


            if next :
            
                block=nn.Sequential(
                    nn.Conv2d(features_in,features_out,kernel_size=3,padding=1),
                    nn.BatchNorm2d(features_out),
                    nn.ReLU(),
                    nn.Conv2d(features_out,features_out,kernel_size=3,padding=1),
                    nn.BatchNorm2d(features_out),
                    nn.ReLU(),
                    nn.Conv2d(features_out,features_out,kernel_size=3,padding=1),
                    nn.BatchNorm2d(features_out),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2,stride=2)
                )
                
            return block

    def forward(self,x : torch.tensor):
        x=self.b1(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        
        x=x.view(x.size(0),-1)
        x=self.l2(self.l1(x))
        
        return self.classifier(x)
    
    
    
###Testing#####|----------------------------------
if __name__=='__main__' :
    x=torch.randn(4,3,224,224,device='cuda')
    model=VGG(3,64).to('cuda')

    out=model(x)

        