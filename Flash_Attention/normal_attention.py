import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def classic_attention(Q : torch.tensor,K: torch.tensor,V:torch.tensor,mask=None):
    scale = 1 /Q.size(-1) ** 0.5
    Q = Q * scale
    attn_weights = torch.einsum('... i d, ... j d -> ... i j', Q, K)

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
    

    attn =F.softmax(attn_weights, dim=-1)
    return attn @ V