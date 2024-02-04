import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_size : int, encoding_dim: int=256, type : str="linear") -> None:
        super().__init__()
        if type=="conv":
            self.encoder=nn.Sequential(
                nn.Conv2d(3,16,3,stride=2,padding=1),
                nn.ReLU(),
                nn.Conv2d(16,32,3,stride=2,padding=1),
                nn.ReLU(),
                nn.Conv2d(32,64,7)
            )
            self.decoder=nn.Sequential(
                nn.ConvTranspose2d(64,32,7),
                nn.ReLU(),
                nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16,3,3,stride=2,padding=1,output_padding=1),
                nn.Sigmoid()
            )
        
        elif type=="linear" : 
            self.encoder=nn.Sequential(
                nn.Linear(input_size,16),
                nn.ReLU(),
                nn.Linear(16,encoding_dim)
            )
            self.decoder=nn.Sequential(
                nn.Linear(encoding_dim,16),
                nn.Sigmoid(),
                nn.Linear(16,input_size),
            )

    
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        
        return x



if __name__=='__main__':
    x=torch.rand(1,3,32,32,requires_grad=True,device="cuda" if torch.cuda.is_available() else "cpu")
    model=AutoEncoder(32,type="conv").to(x.device)
    out=model(x)






















