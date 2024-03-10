import torch
import torch.nn as nn
import torch.nn.functional as F   
from dataclasses import dataclass

@dataclass
class MoeEarg : 
    input_dim : int = 128
    experts : list[nn.Module] =None 
    hidden_dim : int =256 
 


class DummyExpert(nn.Module): 
    def __init__(self, input_dim, out_dim) -> None:
        super().__init__()
        self.fc=nn.Linear(input_dim, out_dim)
    
    def forward(self, x):
        return F.relu(self.fc(x))



##// With softmax a gating signal
class SparseMoE(nn.Module):
    def __init__(self, arg : MoeEarg) -> None:
        super().__init__()
        self.arg=arg
        self.experts=nn.ModuleList(arg.experts)
        self.n_experts=len(arg.experts)
        self.gate=nn.Parameter(torch.randn(arg.input_dim,self.n_experts))
    

    def forward(self, x): 
        raw_gate=torch.einsum("...bnd, ...de->...bne", x, self.gate)
        raw_gate=raw_gate.softmax(dim=-1)
        
        experts_outputs=torch.stack([experts(x) for experts in self.experts],dim=1)
        output=torch.einsum("...bte,...bteo->bto",raw_gate,experts_outputs)

        return output

## // WIth TopK 
class SparseKMoE(nn.Module):
    def __init__(self, arg : MoeEarg, fraction_experts =0.2) -> None:
        super().__init__()
        self.arg=arg
        self.experts=nn.ModuleList(arg.experts)
        self.n_experts=len(arg.experts)
        self.gate=nn.Parameter(torch.randn(arg.input_dim,self.n_experts))
        self.fraction_experts=fraction_experts
    
    def forward(self, x): 
        raw_gate=torch.einsum("...bnd, ...de->...bne", x, self.gate)
        raw_gate=raw_gate.softmax(dim=-1)

        topk_gate_score,topk_gate_indices=raw_gate.topk(int(self.n_experts*self.fraction_experts),dim=2,sorted=False)
        experts_outputs=torch.stack([experts(x) for experts in self.experts],dim=1)
        experts_outputs=experts_outputs.transpose(1,2) #B*NE*T*C->B*T*NE*C

       
        mask=torch.zeros_like(raw_gate).scatter_(2,topk_gate_indices,1)
        gate_topk=mask*raw_gate ## Pluck out the corresponding experts
        gate_topk=F.normalize(gate_topk,dim=2,p=1)
    

        output=torch.einsum("...bte,...bteo->bto",gate_topk,experts_outputs)
        return output
        



