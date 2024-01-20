import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import os
import math
from utils import fast_get_patches
from model import Vit,ViTModelArgs


