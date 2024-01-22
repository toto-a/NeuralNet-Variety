import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from typing import Callable
from tqdm import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt
from _meta import MLP

## Learning to learn 

def reptile(model : nn.Module, nb_iterations: int , sample_task : Callable,
            perform_k_training_steps : Callable, k=1, episilon=0.1):

    
    for _ in tqdm(range(nb_iterations)):
        
        task=sample_task() ##task to sample
        
        ## Training the task on a copy of the model
        ## Unchanging parameters
        phi_tilde = perform_k_training_steps(copy.deepcopy(model), task , k) 

        #Update phi
        with torch.no_grad() :
            for p,g in zip(model.parameters(),phi_tilde):
                p+=episilon * (g-p) ## Update rule 



@torch.no_grad()
def sample_task():
    a=torch.rand(1).item()*4.9 + 0.1 ## a in [0.1, 5.0]
    b=torch.rand(1).item()*2*np.pi  ## b in [0, 2pi]

    x=torch.linspace(-5,5,50)
    y=a*torch.sin(x+b) ## Sample the amplitude

    loss_fct=nn.MSELoss()

    return x,y,loss_fct

    
def perform_k_training_steps(model , task ,k ,batch_size=10):
    optimizer=optim.SGD(model.parameters(), lr=0.02) 

    train_x, train_y, loss_fct= task
    for epoch in range(k*train_x.shape[0]//batch_size):
        ind=torch.randperm(train_x.shape[0])[: batch_size]
        x_batch=train_x[ind].unsqueeze(-1)
        target=train_y[ind].unsqueeze(-1)

        loss= loss_fct(model(x_batch), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ## Return new model weights
    return model.parameters()

    


if __name__=="__main__":
    
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    model=MLP()
    reptile(model, 300, sample_task, perform_k_training_steps)

    with torch.no_grad():

        x=torch.linspace(-5,5,50).unsqueeze(-1)
        y_pred_before=model(x)
        new_task=sample_task()
        true_x, true_y, _ = new_task
        
    ## Perform 32 training steps on the new_task
    perform_k_training_steps(model, new_task, 32)
    y_pred_after=model(x)

    
    plt.plot(x.numpy(), y_pred_before, label='Before')
    plt.plot(x.numpy(), y_pred_after, label ='After')
    plt.plot(true_x.numpy(), true_y.numpy(), label='True')
    plt.legend(fontsize=11)
    plt.savefig('image/Meta_learning.png')
