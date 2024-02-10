import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


## Return the rosenbrock function evaluated at x and y
def rosenbrock(x,y) :
    return (1-x)**2 + 100*(y-x**2)**2


def run_optimization(optimizer_cstm, x, y,n_iter, **optim_kwargs):
    x=torch.tensor(x, requires_grad=True)
    y=torch.tensor(y, requires_grad=True)

    optimizer = optimizer_cstm([x, y], **optim_kwargs)
    path=np.empty((n_iter+1,2))

    path[0,0]=x.item()
    path[0,1]=y.item()

    for i in tqdm(range(1,n_iter+1)):
        optimizer.zero_grad()
        loss = rosenbrock(x,y)
        loss.backward()
        nn.utils.clip_grad_norm_([x,y],1)
        optimizer.step()
        path[i,0]=x.item()
        path[i,1]=y.item()
    
    return path


