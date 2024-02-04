import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from vae import VAE
from tqdm import tqdm



transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = '~/datasets'
train_dataset = MNIST(path, transform=transform, download=True,train=True)
test_dataset  = MNIST(path, transform=transform, download=True,train=False)

# create train and test dataloaders
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train(model, optimizer, epochs, device):
    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, -1).to(device)

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
        return overall_loss



if __name__ == "__main__":
    x_dim = 784
    hidden_dim = 400
    latent_dim = 20
    epochs = 1

    model = VAE(x_dim, hidden_dim, latent_dim, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, optimizer, epochs, device)
    print("Training is done!")

  