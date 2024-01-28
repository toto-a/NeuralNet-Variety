import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import skimage

class Discriminator(nn.Module):
    def __init__(self, img_channels,features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels,features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features, features*2,kernel_size=4, stride=2,padding=1),
            self._block(features*2, features*4,kernel_size=4, stride=2,padding=1),
            self._block(features*4, features*8,kernel_size=4, stride=2,padding=1),
            nn.Conv2d(features*8,1,kernel_size=4,stride=2,padding=0),
            
           
        )
    
    def _block(self,in_chan, out_chan, kernel_size, stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan,kernel_size,stride, padding, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels,features_g):
        super().__init__()
        self.gen = nn.Sequential(  
            #input N*z_dim*1*1
            self._block(z_dim, features_g*16,kernel_size=4), #shape ->N*(fg*16)*4*4
            self._block(features_g*16, features_g*8,kernel_size=4,stride=2,padding=1),
            self._block(features_g*8, features_g*4,kernel_size=4,stride=2,padding=1),
            self._block(features_g*4, features_g*2,kernel_size=4,stride=2,padding=1),
            nn.ConvTranspose2d(features_g*2,img_channels,4,2,1),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )
    def _block(self,in_chan, out_chan, kernel_size, stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan,kernel_size,stride, padding, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(0.2),
        )
        
    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)