import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

from clip_part import ImageEncoder, TextEncoder, ProjectionHead