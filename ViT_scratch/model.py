import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional,Tuple

@dataclass
class ViTModelArgs :
    dim: int =512
    dim_ffn_multiplier : Optional[int] =4
    n_heads : int =6
    dropout : Optional[float] = 0.2
    patch_size : int  = 16 ## 16*16 patches
    img_size: int = None 
    seq_len_img : int =None



class PositionalEmbeddings(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()
        self.position_embedding=nn.Embedding(args.seq_len_img,args.dim)
    
    def forward(self,img_flat:torch.tensor):
        out=self.position_embedding(img_flat)
        return out


class MLPhead(nn.Module):
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


class MHA(nn.Module):
    def __init__(self, args : ViTModelArgs) -> None:
        super().__init__()

        self.n_heads=args.dim

        ##Pass it all in one chunk because they share the same head
        self.wqvk=nn.Linear(args.dim,3*args.dim)
        self.proj=nn.Linear(args.dim,args.dim)
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)

        self.register_buffer('tril',torch.tril(torch.ones(args.seq_len_img,args.seq_len_img))
                             .view(1,1,args.seq_len_img,args.seq_len_img))
    
    def forward(self, emb_img : torch.tensor):

        B,T,C=emb_img.shape
        
        ## (B,T,C)
        q,k,v=self.wqvk(emb_img).split(C,dim=2)

        q=q.view(B,T,self.n_heads,C//self.n_heads).transpose(-2,-3)
        k=k.view(B,T,self.n_heads,C//self.n_heads).transpose(-2,-3)
        v=v.view(B,T,self.n_heads,C//self.n_heads).transpose(-2,-3)

        pre_attn=q@k.transpose(-2,-1)*(1./k.size(-1)) ##to make it unit gaussian
        pre_attn=pre_attn.masked_fill(self.tril[:,:,:T,:T]==0, float('-inf'))
        att=F.softmax(pre_attn,dim=-1)
        att=self.attn_dropout(att)
        y=att@v

        y=att.transpose(1,2).contiguous().view(B,T,C) ##(B,T,C)
        y=self.resid_dropout(self.proj(y))

        return y 


        