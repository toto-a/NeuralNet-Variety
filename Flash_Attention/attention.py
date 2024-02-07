import torch
from torch import nn,einsum
from functools import partial
from torch.autograd.function import Function

from einops import rearrange
from torch.jit import fork, wait

from torch.cuda.amp import autocast,GradScaler
from torch.nn import DataParallel


##Block size : M/4*d (q,k,v,o)
## Load block of size M/4*d into memory 

##Initialize the output matrix O to 0s (N,d)
## l to 0s (N)
## m to (-inf) (N) // the row wise max block
## Divide Q,K,V into M/4d blocks (B*d) each
## Same for O and m

## Outer loop
## Load Ki, Vi from HBM to SRAM
## Inner loop
## Load Qi,Oi,li,mi from HBM to SRAM
## Compute dot product of Qi and Ki.T

## Compute mij(tilde) = rowmax(dot(Qi,Ki.T))
## Pij=exp(dot(Qi,Ki.T)-mij(tilde)) 
## lij(tilde) = rowsum(Pij


class FlashAttentionFunc(Function):
    
    @staticmethod
    @torch.no_grad()
    def forward(ctx, x, k, v, q, mask = None, dropout=None):
        pass

        
