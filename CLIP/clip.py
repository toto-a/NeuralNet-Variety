from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from dataclasses import dataclass
import numpy as np

@dataclass
class CLIPparam :
    att_mask : torch.tensor=None
    transformer_width:int=512
    embed_dim :int =512,
    vision_width :int =768,
    vision_model : str="vit_base_patch16_224",
    context_length :int =77,
    vocab_size : int =49408,
    transformer_width :int =512,
    transformer_heads : int=8,
    transformer_layers : int=12,
        
    

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def forward(self, x:torch.tensor):
        return self.attn(x,x,x,need_weights=False, attn_mask=self.attn_mask)[0]



class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self, dims : CLIPparam) -> None:
        super().__init__()
        self.dims=dims
