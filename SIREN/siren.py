import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt




class SineLayer(nn.Module):
    def __init__(self, w0) -> None:
        super().__init__()
        self.w0=w0
    def forward(self, x):
        return torch.sin(self.w0*x)

class SIREN(nn.Module):
    def __init__(self,w0=30, in_dim=2, hidden_dim=256, out_dim=1 ) -> None:
        super().__init__()

        ##w0 frequencies
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim,hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim,hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim,hidden_dim), SineLayer(w0),
            nn.Linear(hidden_dim,out_dim)
        )

        
        ##Init the weights
        with torch.no_grad():
            self.net[0].weight.uniform_(-1/in_dim,1/in_dim)
            self.net[2].weight.uniform_(-np.sqrt(6./hidden_dim)/w0,
                                                                    np.sqrt(6./hidden_dim)/w0)
            self.net[4].weight.uniform_(-np.sqrt(6./hidden_dim)/w0,
                                                                    np.sqrt(6./hidden_dim)/w0)
            self.net[6].weight.uniform_(-np.sqrt(6./hidden_dim)/w0,
                                                                    np.sqrt(6./hidden_dim)/w0)
            self.net[8].weight.uniform_(-np.sqrt(6./hidden_dim)/w0,
                                                                    np.sqrt(6./hidden_dim)/w0)
            
    
    def forward(self,x ):
        return self.net(x)
        

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1 ) -> None:
        super().__init__()

        ##w0 frequencies
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )

        
            
    
    def forward(self,x ):
        return self.net(x)




    