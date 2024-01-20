import torch
import torch.nn as nn
import torch.nn.functional as F 
import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


device='cuda' if torch.cuda.is_available() else 'cpu'

class Nerf(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128) -> None:
        super().__init__()

        self.block1=nn.Sequential(nn.Linear(embedding_dim_pos*6 +3 , hidden_dim),nn.ReLU(),
                                  nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                  nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                  nn.Linear(hidden_dim,hidden_dim),nn.ReLU())
        
        self.block2=nn.Sequential(nn.Linear(embedding_dim_pos*6 +hidden_dim+3 , hidden_dim),nn.ReLU(),
                                  nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                  nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                  nn.Linear(hidden_dim,hidden_dim+1),nn.ReLU())

        self.block3=nn.Sequential(nn.Linear(embedding_dim_direction*6 + hidden_dim + 3, hidden_dim//2, nn.ReLU()))
        self.block4=nn.Sequential(nn.Linear(hidden_dim//2,3),nn.Sigmoid())

        self.embedding_dim_pos=embedding_dim_pos
        self.embedding_dim_direction=embedding_dim_direction
        self.relu=nn.ReLU()
    
    @staticmethod
    def positional_encodings(x:torch.tensor,L):
        out=[x]
        for j in range(L):
            out.append(torch.sin(2**j*x))
            out.append(torch.cos(2**j*x))
        return torch.cat(out,dim=1)
    
    def forward(self, o : torch.tensor, d: torch.tensor):
        emb_x=self.positional_encodings(o,self.embedding_dim_pos) ##position
        emb_d=self.positional_encodings(d,self.embedding_dim_direction ##direction
                                    )
        h=self.block1(emb_x)
        tmp=self.block2(torch.cat((h,emb_x),dim=1))
        h,sigma=tmp[:,:-1], self.relu(tmp[:,:-1])
        h=self.block3(torch.cat((h,emb_d),dim=1))
        c=self.block4(h)

        return c,sigma



def compute_transmittance(alphas):
    accumulated_transmittance=torch.cumprod(alphas,1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0],1),device=alphas.device), 
                      accumulated_transmittance[:,:-1]),dim=-1)


def render_rays(nerf_model, ray_orgins, ray_direction, hn=0, hf=0.5, nb_bins=192):
    
    t=torch.linspace(hn,hf,nb_bins,device=device).expand(ray_orgins.shape[0], nb_bins)
    
    #Perturb sampling along each ray//randomly
    mid=(t[:,:-1] + t[:,1:])/2
    lower=torch.cat((t[:,:1], mid), dim=-1)
    upper=torch.cat((t[:,-1:], mid), dim=-1)
    u=torch.rand(t.shape,device=device)
    t=lower + (upper-lower)*u
    
    delta=torch.cat((t[:,1:] - t[:,:-1],torch.tensor([1e10],device=device).expand(ray_orgins.shape[0],1)),-1)
    x=ray_orgins.unsqueeze(1) + t.unsqueeze(2)*ray_direction.unsqueeze(1) #(batch_size, nb_bins, 3)
    ray_direction=ray_direction.expand(nb_bins, ray_direction.shape[0], 3).transpose(0,1) ##Position of the rays at a given steps t

    colors,sigma=nerf_model(x.reshape(-1,3),ray_direction.reshape(-1,3))
    colors=colors.reshape(x.shape)
    sigma=sigma.reshape(x.shape[:-1])
    
    alpha=1-torch.exp(-sigma.reshape(x.shape[:-1])*delta)
    weights=compute_transmittance(1-alpha).unsqueeze(2)*alpha.unsqueeze(2)
    c=(weights*colors).sum(dim=1) ##Pixel values
    weight_sum=weights.sum(-1).sum(-1) ## Regularization for white background

    return c+1 -weight_sum.unsqueeze(-1)

    
def train(nerf_model, optimizer, scheduler, data_loader, device=device, hn=0, hf=1, nb_epochs=int(1e5), nb_bins=192, H=400, W=400):

        training_loss=[]
        for _ in tqdm(nb_epochs):
            for batch in data_loader :
                
                ray_origins=batch[:,:3].to(device)
                ray_directions= batch[:,3:6].to(device)
                ground_truth_px_values= batch[:,6:].to(device)
                
                regenerate_pixel_values=render_rays(nerf_model=nerf_model, ray_orgins=ray_origins, ray_direction=ray_directions,
                                                    hn=hn,hf=hf, nb_bins=nb_bins)
                loss=((ground_truth_px_values-regenerate_pixel_values)**2).sum()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())

            scheduler.step()

            ##Sample after each epochs
            for img_index in range(200) :
                test(hn,nerf_model,hf,img_index,nb_bins, H=H, W=W)
        
        return training_loss     


@torch.no_grad()
def test(hn, model,hf, dataset,chunk_size=10, img_index=0, nb_bins=192, H=400 ,W=400):

    ray_orgins=dataset[img_index*H*W:(img_index+1)*H*W,:3]
    ray_directions=dataset[img_index*H*W:(img_index+1)*H*W,3:6]
    
    data=[]
    for i in range(int(np.ceil(H/chunk_size))):
        ray_orgins_=ray_orgins[i*chunk_size*W:(i+1)*chunk_size*W,:3]
        ray_directions_=ray_directions[i*chunk_size*W:(i+1)*chunk_size*W,:3]

        regenerated_px_values=render_rays(model,ray_orgins_,ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img=torch.cat(data).data.cpu().numpy().reshape(H,W,3)
    plt.figure()
    plt.imshow(img)
    plt.save_fig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()
    
    
    


    