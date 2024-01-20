import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional,Tuple

from utils import fast_get_patches

@dataclass
class ViTModelArgs :
    dim: int =512
    dim_ffn_multiplier : Optional[int] =4
    n_heads : int =6
    n_layers : int =12
    dropout : Optional[float] = 0.2
    patch_size : int  = 16 ## 16*16 patches
    img : torch.Tensor =None
    img_size: int = None 
    out_classes : int = 10
    seq_len_patches : int =patch_size**2
    hidden_d_mapper_mult : float = 0.5 
    batch_size : int =32
    device : str='cuda' if torch.cuda.is_available() else 'cpu'



##Learnable positional encoding
class PositionalEmbeddings(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()
        self.seq=args.seq_len_patches + 1
        self.hidden_d_mapper=args.hidden_d_mapper_mult*args.seq_len_patches
        self.position_embedding=nn.Embedding(args.seq_len_patches,self.hidden_d_mapper)
    
    def forward(self,img_flat:torch.tensor):
        out=img_flat+self.position_embedding(img_flat)
        return out


class MLP(nn.Module):
    def __init__(self,args:ViTModelArgs) -> None:
        super().__init__()
        self.dim=args.dim
        self.hidden_dim=args.dim_ffn_multiplier*self.dim
        self.net=nn.Sequential(
            nn.Linear(args.dim,self.hidden_dim,bias=False),
            nn.GELU(),
            nn.Linear(args.dim,self.hidden_dim,bias=False),
            nn.Linear(self.hidden_dim,args.img_size,bias=False),
            nn.Dropout(args.dropout)
        )
    
    def forward(self,img_embd : torch.tensor):
        x=self.net(img_embd)
        return x


##Self attention
class MHA(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()

        self.dim=args.dim
        self.n_heads=args.n_heads

        ##Pass it all in one chunk because they share the same head
        self.wqvk=nn.Linear(args.dim,3*args.dim)
        self.proj=nn.Linear(args.dim,args.dim)
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

        y=att.transpose(1,2).contiguous().view(B,T,C) ##(B,T,C)
        y=self.resid_dropout(self.proj(y))

        return y 


class EncoderBlock(nn.Module):
    def __init__(self,args : ViTModelArgs) -> None:
        super().__init__()
        self.n_heads=args.n_heads
        self.pre_attn_norm=nn.LayerNorm(args.seq_len_img)
        self.sa_heads=MHA(args=args)
        self.pre_mlp_norm=nn.LayerNorm(args.dim)
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
        self.l1=nn.Linear(args.dim,args.out_classes)
    
    def forward(self, encoder_output: torch.tensor):
        return F.softmax(self.l1(encoder_output),dim=-1)
        
        
        


class Vit(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()
        self.patch_dim=(args.patch_size,args.patch_size)
        self.device=args.device
        self.hidden_d_mapper=args.hidden_d_mapper_mult*args.seq_len_patches
        
        ### Class Token and linear mapper
        self.class_token=nn.Parameter(torch.rand(1,self.hidden_d_mapper))
        self.linear_mapper=nn.Linear(args.seq_len_patches,self.hidden_d_mapper)

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
        self.class_token=self.class_token[None,:,:].expand(B,-1,s_patches_mapped)

        #(Batch_size,n_patches,hidden_d) and (Batch_size,1,hidden_d)-> (B,n_patches+1,hidden_d)
        tokens=torch.cat([self.class_token,tokens],dim=1)
        
        ##Repeat the positional encodings for each tokens in the batch
        self.position_embedding=self.position_embedding.repeat(B,1,1)
        out_embed=self.position_embedding(tokens)
        out_encoder=self.encoder(out_embed)
        out_classes=self.classifier(out_encoder)

        return out_classes