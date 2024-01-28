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
from dcgan import Discriminator,Generator,initialize_weights

# Hyperparameters 
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
batch_size = 32
z_dim = 100
IMAGE_SIZE=64
img_channels=1
num_epochs = 5
features_g=64
features=64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(img_channels)],[0.5 for _ in range(img_channels)]),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

###Set up the generator and the discriminator
gen=Generator(z_dim,img_channels,features_g).to(device)
initialize_weights(gen)
disc=Discriminator(img_channels,features).to(device)
initialize_weights(disc)

optim_disc=optim.Adam(disc.parameters(),lr=lr, betas=(0.5,0.999))
optim_gen=optim.Adam(gen.parameters(),lr=lr,betas=(0.5,0.999))
criterion=nn.BCELoss()

###Noise for testing purpose
fixed_noise = torch.randn((32, z_dim,1,1)).to(device)

writer_fake=SummaryWriter(f"logs/fake")
writer_real=SummaryWriter(f"logs/real")
step=0


for epoch in tqdm(range(num_epochs)) :
    for batch_idx, (real,_) in enumerate(loader): 
        real=real.to(device) ##flatten
        noise = torch.randn(batch_size, z_dim,1,1).to(device)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True) 
        optim_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).reshape(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optim_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
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