import torch
import numpy as np
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1) -> None:
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim), 

            
        )
    
    def forward(self,noise):
        return self.net(noise)
    