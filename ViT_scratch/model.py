import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional,Tuple

from utils import fast_get_patches

@dataclass
class ViTModelArgs :
    hidden_size: int =48
    dim_ffn_multiplier : Optional[int] =4
    n_heads : int =6
    n_layers : int =12
    dropout : Optional[float] = 0.2
    img_size : int =None
    patch_size : int  = 4 ## 4*4 patches
    n_channels : int = 1
    out_classes : int = 10
    seq_input_patches : int =patch_size**2 * n_channels
    seq_len_patches : int =1 ##n_patches+1
    patches_vocab_size : int = seq_len_patches *seq_input_patches
    batch_size : int =32
    n_epochs : int =100
    device : str='cuda' if torch.cuda.is_available() else 'cpu'



##Learnable positional encoding
class PositionalEmbeddings(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()
        self.seq_len=args.seq_len_patches
        self.hidden_d_mapper=args.hidden_size
        self.hidden_size=args.hidden_size
        self.position_embedding=nn.Parameter(torch.randn(1,self.seq_len +1 , self.hidden_size))
    
    def forward(self,x):
        return x+self.position_embedding


class MLP(nn.Module):
    def __init__(self,args:ViTModelArgs) -> None:
        super().__init__()
        self.dim=args.hidden_size
        self.hidden_dim=args.dim_ffn_multiplier*self.dim
        self.net=nn.Sequential(
            nn.Linear(self.dim,self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim,self.dim),
            nn.Dropout(args.dropout)
        )
    
    def forward(self,img_embd : torch.tensor):
        x=self.net(img_embd)
        return x


##Self attention
class MHA(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()

        self.dim=args.hidden_size
        self.n_heads=args.n_heads

        ##Pass it all in one chunk because they share the same head
        self.wqvk=nn.Linear(self.dim,3*self.dim,bias=False)
        self.proj=nn.Linear(self.dim,self.dim)
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)

    
    def forward(self, emb_img : torch.tensor):

        B,T,C=emb_img.shape
        
        ## (B,T,C)
        q,k,v=self.wqvk(emb_img).split(C,dim=2)

        q=q.view(B,T,self.n_heads,C//self.n_heads).transpose(-2,-3)
        k=k.view(B,T,self.n_heads,C//self.n_heads).transpose(-2,-3)
        v=v.view(B,T,self.n_heads,C//self.n_heads).transpose(-2,-3)

        pre_attn=q@k.transpose(-2,-1)*(1./k.size(-1)) ##to make it unit gaussian
        att=F.softmax(pre_attn,dim=-1)
        att=self.attn_dropout(att)
        y=att@v

        y=y.transpose(1,2).contiguous().view(B,T,C) ##(B,T,C)
        y=self.resid_dropout(self.proj(y))

        return y 


class EncoderBlock(nn.Module):
    def __init__(self,args : ViTModelArgs) -> None:
        super().__init__()
        self.n_heads=args.n_heads
        self.hidden_size=args.hidden_size
        self.pre_attn_norm=nn.LayerNorm(self.hidden_size)
        self.sa_heads=MHA(args=args)
        self.pre_mlp_norm=nn.LayerNorm(args.hidden_size)
        self.mlp_head=MLP(args=args)
    
    def forward(self, embed_patches : torch.tensor):
        _x=embed_patches
        embed_patches=self.pre_attn_norm(embed_patches)
        out=self.pre_mlp_norm(self.sa_heads(embed_patches)+_x)
        out=out + self.mlp_head(out)
        return out



class VitEncoder(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()
        self.n_layers=args.n_layers
        self.encoders=nn.ModuleList([EncoderBlock(args) for _ in range(self.n_layers)])
    
    def forward(self,x:torch.tensor):

        for encoder in self.encoders:
            x=encoder(x)

        return x

class Classifier(nn.Module):
    def __init__(self, args:ViTModelArgs ) -> None:
        super().__init__()
        self.l1=nn.Linear(args.hidden_size,args.out_classes)
    
    def forward(self, encoder_output: torch.tensor):
        return F.softmax(self.l1(encoder_output),dim=-1)
        
        
        


class Vit(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()
        self.patch_dim=(args.patch_size,args.patch_size)
        self.device=args.device
        self.dim=args.hidden_size
        
        ### Class Token and linear mapper
        self.class_token=nn.Parameter(torch.rand((1,1,self.dim),dtype=torch.float,device=self.device))
        self.linear_mapper=nn.Linear(args.seq_input_patches,self.dim)

        ###Positional embeddings and token embeddings
        self.position_embedding=PositionalEmbeddings(args)
        self.encoder=VitEncoder(args=args)
        self.classifier=Classifier(args)
    
    
    def forward(self, imgs : torch.tensor):


        patches=fast_get_patches(imgs,self.patch_dim,self.device)
        tokens=self.linear_mapper(patches)

        ##Add classification tokens
        B,n_patches,s_patches_mapped=tokens.shape

        #(1,hidden_d)->(Batch_size,1,hidden_d)
        cls_token=self.class_token.expand(B,-1,s_patches_mapped)

        #(Batch_size,n_patches,hidden_d) and (Batch_size,1,hidden_d)-> (B,n_patches+1,hidden_d)
        tokens=torch.cat([cls_token,tokens],dim=1)
        
        ##Sum the encodings
        out_embed=self.position_embedding(tokens)
        out_encoder=self.encoder(out_embed)

        out_cls_token=out_encoder[:,0]
        out_classes=self.classifier(out_cls_token)

        return out_classes
    
    def _init_weight(self,module) :
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        