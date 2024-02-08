import torch
from torch import nn,einsum
from functools import partial
from torch.autograd.function import Function
from einops import rearrange,repeat

from torch.cuda.amp import autocast,GradScaler
from torch.nn import DataParallel


##Assume Block size
Bc=16

###-------------

##Block size : M/4*d (q,k,v,o)
## Load block of size M/4*d into memory 

##Initialize the output matrix O to 0s (N,d)
## l to 0s (N) // the sum of the row wise exp block
## m to (-inf) (N) // the row wise max block
## Divide Q,K,V into M/4d blocks (B*d) each
## Same for O and m

## Outer loop (Tr)
## Load Ki, Vi from HBM to SRAM
## Inner loop (Tc)
## Load Qi,Oi,li,mi from HBM to SRAM
## Compute dot product of Qi and Ki.T

## Compute mij(tilde) = rowmax(dot(Qi,Ki.T))
## Pij=exp(dot(Qi,Ki.T)-mij(tilde)) 
## lij(tilde) = rowsum(Pij)

def exists(val):
    return val is not None

class FlashAttentionFunc(Function):
    
    @staticmethod
    @torch.no_grad()
    def forward(K, v, q, mask = None, dropout=None):
        O=torch.zeros_like(q)
        l=torch.zeros_like(q[:-1])
        m=torch.full_like(q[:-1],-float('inf'))
        
        
        Br=min(Bc,q.size(-2)) #number of heads and Bc
        
        if exists(mask) and mask.dim() == 2:
            mask = rearrange(mask,'b n -> b 1 1 n') # (B,1,1,N)

        mask_block=mask.split(Br, dim=-1)

        q_block=q.split(Br, dim=-2)
        v_block=v.split(Bc, dim=-2)
        k_block=K.split(Bc, dim=-2)
        o_block=O.split(Br, dim=-2)

        m_block=list(m.split(Br, dim=-2)) # 1D
        l_block=list(l.split(Br, dim=-2)) # 1D

        Tr=len(q_block)
        Tc=len(k_block)

        for j in range(Tc) :
            K_j=k_block[j]
            V_j=v_block[j]
            mask_j=mask_block[j]


            for i in range(Tr) :
                Q_i=q_block[i]
                O_i=o_block[i]
                l_i=l_block[i]
                m_i=m_block[i]
                
                scale=1/(Q_i.size(-1)**0.5) ##Scaling factor to make it unit gaussian
                Q_i=Q_i*scale

                sij=einsum('bnid,bnjd->bnij',Q_i,K_j) ## Dot product of Q_i and K_j.T

                ##Mask 
                sij=sij.masked_fill(mask_j==0,'-inf')

                m_tilde_ij=torch.max(sij,dim=-1).values
                P_tilde_ij=torch.exp(sij-m_tilde_ij)
                l_tilde_ij=torch.sum(P_tilde_ij,dim=-1)

                m_inew=torch.max(m_i,m_tilde_ij)
                l_inew=torch.exp(m_i-m_inew)*l_i + torch.exp(m_tilde_ij-m_inew)*l_tilde_ij

                O_i=torch.inverse(torch.eye(l_inew) )* (torch.eye(l_i)*torch.exp(m_i-m_inew)*O_i + torch.exp(m_tilde_ij-m_inew)*l_tilde_ij )
                l_i=l_inew
                m_i=m_inew
    
        return O 
                


        
##### Debugging purposes 
if __name__ == "__main__":
    Q = torch.randn(1, 2, 1024, 1024, requires_grad=True)
    V = torch.randn(1, 2, 1024, 1024, requires_grad=True)
    K = torch.randn(1, 2, 1024, 1024, requires_grad=True)
    mask = torch.randint(0, 2, (1, 4096))

    d=FlashAttentionFunc.apply(K, V, Q, mask)
        
        

        
