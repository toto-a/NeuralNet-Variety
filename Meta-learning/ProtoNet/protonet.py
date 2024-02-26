import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as m


## ProtoNet implementation

### Create a DenseNet 
def get_convnet(output_size):
    conv_net=m.DenseNet(
        growth_rate=32,
        block_config=(6,6,6,6),
        bn_size=2,
        num_init_features=64,
        num_classes=output_size
        )

