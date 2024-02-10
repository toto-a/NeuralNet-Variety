import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer


class MyOptim(Optimizer):
    def __init__(self, params, lr=0.001):
        defaults = dict(lr=lr)
        super(MyOptim, self).__init__(params, defaults)
    
    def step(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure()
        
        if not self.state :
            self.state["step"]=1
        else: 
            self.state["step"]+=1


        c=1
        if self.state["step"]%100==0:
            c=50
        
        grad=None
        ## Loop while trying to find a tensor with a gradient
        while grad is None:
            params=np.random.choice(self.param_groups)
            perm=torch.randperm(len(params["params"]))[:1]
            tensor=params["params"][perm]
            grad=tensor.grad.data
        

        ### Create a mask 
        ix=np.random.randint(tensor.numel())
        mask_flat=np.zeros(tensor.numel())
        mask_flat[ix]=1
        mask=mask_flat.reshape(tensor.shape)


        tensor.data.add_(grad*mask,alpha=-params["lr"]*c)

        return loss