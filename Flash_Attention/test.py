import torch
from torch.func import jacrev

from flash_attention import FlashAttention
from normal_attention import classic_attention


Q = torch.randn(1, 1, 2048, 512, requires_grad=True)
K = torch.randn(1, 1, 2048, 512, requires_grad=True)
V = torch.randn(1, 1, 2048, 512, requires_grad=True)
mask = torch.randint(0, 2, (1, 2048))

def loss_fn(fn, *args):
    return torch.sum(fn(*args))

args = (Q, K, V, mask) 