
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm,trange

# from gan import Discriminator,Generator
from wgan import Discriminator,Generator,initialize_weights
from utils import gradient_penalty

# Hyperparameters 
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
batch_size = 64
z_dim = 100
IMAGE_SIZE=64
img_channels=3
num_epochs = 5
features_g=64
features=64
CRITIC_ITERATION=5
LAMBDA=10
# CLIP=0.01

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(img_channels)],[0.5 for _ in range(img_channels)]),
    ]
)

dataset=datasets.ImageFolder(root="celeb_datataset",transform=transforms)
dataloader=DataLoader(dataset, batch_size=batch_size,shuffle=True)


###Set up the generator and the discriminator
gen=Generator(z_dim,img_channels,features_g).to(device)
initialize_weights(gen)
critic=Discriminator(img_channels,features).to(device)
initialize_weights(critic)

# optim_critic=optim.RMSprop(critic.parameters(),lr=lr)
optim_critic=optim.Adam(critic.parameters(),lr=lr, betas=(0.,0.9))
# optim_gen=optim.RMSprop(gen.parameters(),lr=lr)
optim_gen=optim.Adam(gen.parameters(),lr=lr,betas=(0.,0.9))

###Noise for testing purpose
fixed_noise = torch.randn((32, z_dim,1,1)).to(device)

writer_fake=SummaryWriter(f"logs/fake")
writer_real=SummaryWriter(f"logs/real")
step=0


for epoch in tqdm(range(num_epochs)) :
    for batch_idx, (real,_) in enumerate(dataloader): 
        real=real.to(device) ##flatten

        ## The discriminator trains CRITIC_ITERATION more than the generator
        for _ in range(CRITIC_ITERATION):
            noise = torch.randn(batch_size, z_dim,1,1).to(device)
            fake = gen(noise) 
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp=gradient_penalty(critic,real,fake,device)
            loss_critic=(
                -(torch.mean(critic_real) - torch.mean(critic_fake)) 
                + LAMBDA*gp
                )##minimizing
            critic.zero_grad()
            loss_critic.backward(retain_graph=True) 
            critic.step()

            # ##Clamp the gradient
            # for p in critic.parameters():
            #     p.data.clamp_(-CLIP,CLIP)
            
        
        ## Train the generator minimize -E[f(z)]  
        output = critic(fake).reshape(-1)
        lossG = -torch.mean(output)
        gen.zero_grad()
        lossG.backward()
        optim_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1