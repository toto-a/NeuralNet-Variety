import torch
import torch.nn as nn
import torch.nn.functional as F





def get_patches(img : torch.tensor, patch_size : int):
    flat_img=img.flatten()
    assert flat_img.size%patch_size==0
    
    flattened_patches=flat_img[::patch_size]
    return flattened_patches

    

    
