import torch
import math
import torch.nn as nn
from typing import Tuple


##Patch size represent the size of the patch we want 16*16 /  32*32 / 4*4
def fast_get_patches(imgs : torch.tensor, patch_dim : Tuple[int], device : str):
    
    B,c,h,w=imgs.shape
    assert h==w, "Only working on square image"
    assert patch_dim[0]==patch_dim[1], "Only square patches are accepted"
    assert patch_dim[0]>0, "Positive number accepted only"
    assert len(patch_dim)==2
    patch_size=patch_dim[0]*patch_dim[1]
    
    num_patches=(h//patch_dim[0])**2
    ## (Batch_size, #Patches , Patch size)
    patches=imgs.view(B,num_patches,patch_size*c).type_as(imgs).to(device)
    
    return patches


def slow_get_patches(imgs : torch.tensor, patch_dim : Tuple[int], device :str):
    
    B,c,h,w=imgs.shape
    assert h==w, "Only working on square image"
    assert patch_dim[0]==patch_dim[1], "Only square patches are accepted"
    assert patch_dim[0]>0, "Positive number accepted only"
    assert len(patch_dim)==2
    patch_size=patch_dim[0]*patch_dim[1]
    
    num_patches=(h//patch_dim[0])**2
    num_patch=int(h//patch_dim[0])
    patches=torch.zeros(B,num_patches,patch_size*c).type_as(imgs)
    
    for idx,img in enumerate(imgs):
        for i in range(num_patch):
            for j in range(num_patch) :
                
                ##Each patch_size 
                patch=img[:,i*patch_dim[0]:(i+1)*patch_dim[0],j*patch_dim[0]:(j+1)*patch_dim[1]]
                patches[idx,i*num_patch+j]=patch.flatten()
    
    patches=patches.to(device)
    return patches


class PatchEmbedddings(nn.Module) :
    def __init__(self,config):
        super().__init__()
        
        self.image_size=config.img_size
        self.patch_size = config.patch_size
        self.num_channels = config.n_channels
        self.hidden_size = config.hidden_size
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, H, W) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
        