import torch
import torch.nn


def gradient_penalty(critic, real, fake, device):
    B,C,H,W=real.shape
    epsilon=torch.rand(B,1,1,1).repeat(1,C,H,W).to(device)
    interpolated_image=epsilon*real + (1-epsilon)*fake
    
    #calculate the critic score
    mixed_score=critic(interpolated_image)
    
    gradient=torch.autograd.grad(
        inputs=interpolated_image,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient.view(gradient.shape[0],-1)
    gradient_norm=gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    
    return gradient_penalty
    
    
    
    
    
