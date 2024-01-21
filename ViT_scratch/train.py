import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from model import Vit,ViTModelArgs


###-----------------Model Args------------------###
hidden_size=48
dim_ffn_multiplier =4
n_heads : int =6
n_layers : int =12
dropout : float = 0.2
patch_size : int  = 4 ## 4*4 patches
img_size: int = 28
n_channels : int =1
out_classes : int = 10
seq_input_patches : int =patch_size**2 * n_channels
seq_len_patches : int =(img_size//patch_size)**2 
patches_vocab_size : int = seq_len_patches *seq_input_patches
batch_size : int =32
n_epochs : int =100
device : str='cuda' if torch.cuda.is_available() else 'cpu'
LR=5e-3 

model_args=dict(hidden_size=hidden_size,dim_ffn_multiplier=dim_ffn_multiplier, n_heads=n_heads, n_layers=n_layers,
                dropout=dropout, patch_size=patch_size,n_channels=n_channels, out_classes=out_classes,
                seq_input_patches=seq_input_patches,seq_len_patches=seq_len_patches,patches_vocab_size=patches_vocab_size, batch_size=batch_size, n_epochs=n_epochs,
                device=device
                )

vitargs=ViTModelArgs(**model_args)
model=Vit(vitargs)

###-----------------Model Args------------------###


@torch.no_grad()
def test(test_loader, model, criterion): 
    correct, total = 0, 0
    test_loss = 0.0
    for batch in tqdm(test_loader, desc="Testing"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        test_loss += loss.detach().cpu().item() / len(test_loader)

        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total += len(x)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")


def main():
   # Loading data
    model.to(device)
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in trange(n_epochs, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.2f}")
    test(test_loader=test_loader,model=model,criterion=criterion)


if __name__=="__main__":
    main()