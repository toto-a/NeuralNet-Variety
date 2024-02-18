import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel,DistilBertConfig

import config as cfg

class ImageEncoder(nn.Module):
    def __init__(self, model_name=cfg.model_name, pretrained=cfg.pretrained, trainable=cfg.trainable) :
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        for p in self.model.parameters():
            p.requires_grad = trainable
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class TextEncoder(nn.Module):
    def __init__(self, model_name=cfg.text_encoder_model, pretrained=cfg.pretrained, trainable=cfg.trainable):
        super().__init__()
        
        if pretrained :
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            config = DistilBertConfig()
            self.model = DistilBertModel(config)
        

        for p in self.model.parameters():
            p.requires_grad = trainable
        
        ### CLS token 
        self.cls_token= 0
    
    def forward(self,inputs: torch.Tensor, attention_mask) -> torch.Tensor:
        return self.model(inputs,attention_mask=attention_mask).last_hidden_state[:,self.cls_token,:]
            

class ProjectionHead(nn.Module) :
    def __init__(self, embed_dim, proj_dim=cfg.proj_dim, dropout=cfg.dropout) :
        super().__init__()

        self.proj=nn.Linear(embed_dim,proj_dim)
        self.gel=nn.GELU()
        self.fc=nn.Linear(proj_dim,proj_dim)
        self.dropout=nn.Dropout(dropout)
        self.layernorm=nn.LayerNorm(proj_dim)
    

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        proj=self.proj(x)
        x=self.gel(proj)
        x=self.fc(x)
        x=self.dropout(x)
        x=x+proj
        x=self.layernorm(x)
        return x


        

                    