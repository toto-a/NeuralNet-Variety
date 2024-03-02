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
    embed_dim :int =512
    vision_width :int =768
    vision_model : str="vit_base_patch16_224"
    context_length :int =77
    vocab_size : int =49408
    transformer_width :int =512
    transformer_heads : int=8
    transformer_layers : int=12
        
    

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


        self.visual_model=timm.create_model(self.dims.vision_model,num_classes=0)
        self.transformer=Transformer(
            width=dims.transformer_width,
            layers=dims.transformer_layers,
            heads=dims.transformer_heads,
            attn_mask=self.build_attn_mask(),
        )

        self.token_embedding=nn.Embedding(dims.vocab_size,dims.embed_dim)
        self.position_embd=nn.Parameter(torch.empty(
            dims.context_length, dims.embed_dim
        ))

        self.lm_final=nn.LayerNorm(dims.transformer_width)

        self.image_proj=nn.Parameter(
            torch.empty(dims.vision_width,dims.embed_dim)
        )


        self.text_proj=nn.Parameter(
            torch.empty(dims.transformer_width,dims.embed_dim)
        )
    
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_params()
    

    def initialize_params(self) :

        nn.init.normal_(self.token_embedding.weight, std=0.02)

        ## Position embedding are less important than token embedding

        nn.init.normal_(self.position_embd, std=0.01)

        proj_std=(self.dims.transformer_width**-0.5) *(
            (2*self.dims.transformer_layers)**-0.5
        )

        attn_std=self.dims.transformer_width**-0.5
        fc_std=(2*self.dims.transformer_layers)**-0.5

        for block in self.transformer.resblocks :
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_proj, std=self.dims.vision_width**-0.5)
        nn.init.normal_(self.text_proj, std=self.dims.transformer_width**-0.5)
    

    def build_attn_mask(self): 

        mask = torch.empty(self.dims.context_length, self.dims.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask



    def encode_image(self,image) :
        return self.visual_model(image)@self.image_proj
    

    def encode_text(self,text) :
        x=self.token_embedding(text)
        x=x+self.position_embd
        x=x.permute(1,0,2)
        x=self.transformer(x)
        x=x.permute(1,0,2)

        x=self.lm_final(x)

        return x[torch.arange(x.size(0)),torch.argmax(x,-1)]
    

    def forward(self,text, image):
        text_features=self.encode_text(text)
        image_features=self.encode_image(image)

        ## Normalized features
        text_features=text_features/text_features.norm(dim=1,keepdim=True)
        image_features=image_features/image_features.norm(dim=1,keepdim=True)


        ## Scale
        self.scale=self.logit_scale.exp()
        logits_per_image=self.scale*image_features@text_features.T 
        logits_per_text=self.scale*text_features@image_features.T

        return logits_per_image,logits_per_text




        



