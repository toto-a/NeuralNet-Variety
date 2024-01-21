import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm,trange
import skimage

from siren import SIREN,MLP

EPOCHS=300

def train(model,model_optimizer, nb_epochs=EPOCHS):
    psnr=[]
    for _ in tqdm(range(nb_epochs)) : 
        model_out=model(pixel_coordinates)
        loss=((model_out-pixels_values)**2).mean()
        psnr.append(20*np.log10(1.0/np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
    
    return psnr, model_out



if __name__=="__main__" :
    device='cuda'
    siren=SIREN().to(device)
    mlp=MLP().to(device)
    
    ##Target
    img=((torch.from_numpy(skimage.data.camera()- 127.5))/127.5) ##normalize -1 and 1
    pixels_values=img.reshape(-1,1).to(device)

    
    ##Input
    resolution=img.shape[0]
    tmp=torch.linspace(-1,1,steps=resolution)
    x,y=torch.meshgrid(tmp,tmp)
    pixel_coordinates=torch.cat((x.reshape(-1,1),y.reshape(-1,1)),dim=-1).to(device)

    fig,axes=plt.subplots(1,5,figsize=(15,3))
    axes[0].imshow(img,cmap='gray')
    axes[0].set_title('Ground truth',fontsize=15)

    for i, model in enumerate([mlp,siren]) :
        
        optim_=optim.Adam(lr=1e-4, params=model.parameters())
        psnr, model_output =train(model, optim_)
        
        ##PLot the results
        axes[i+1].imshow(model_output.cpu().view(resolution,resolution).detach().numpy(),cmap='gray')
        axes[i+1].set_title('ReLU' if (i==0) else 'SIREN', fontsize=15)

        axes[4].plot(psnr,label='ReLU' if (i==0) else 'SIREN', c='C0' if (i==0) else 'mediumseagreen')
        axes[4].set_xlabel('Iterations',fontsize=16)
        axes[4].set_ylabel('PSNR', fontsize=16)
        axes[4].legend(fontsize=15)
    
    for i in range(4) :
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    axes[3].axis('off')
    plt.savefig(f'Siren_out_{EPOCHS}.png')

        