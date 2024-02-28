from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


