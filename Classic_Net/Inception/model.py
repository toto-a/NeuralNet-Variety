import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        return x    

class BlockInception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(BlockInception, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1pool, kernel_size=1)
        )
    
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


    
class AuxInception(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxInception, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = self.batchnorm(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
 

class InceptionNet(nn.Module):
    def __init__(self, aux_logits=True, n_classes=1000) -> None:
        super().__init__()
        self.aux_logits = aux_logits
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)

        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool=nn.MaxPool2d(3, stride=2, padding=1)
        self.a3 = BlockInception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = BlockInception(256, 128, 128, 192, 32, 96, 64)
        self.a4= BlockInception(480, 192, 96, 208, 16, 48, 64)
        self.b4= BlockInception(512, 160, 112, 224, 24, 64, 64)
        self.c4= BlockInception(512, 128, 128, 256, 24, 64, 64)
        self.d4= BlockInception(512, 112, 144, 288, 32, 64, 64)
        self.e4= BlockInception(528, 256, 160, 320, 32, 128, 128)
        self.a5= BlockInception(832, 256, 160, 320, 32, 128, 128)
        self.b5= BlockInception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.4)

        self.fc=nn.Linear(1024, n_classes)

        if self.aux_logits:
            self.aux1 = AuxInception(512, n_classes)
            self.aux2 = AuxInception(528, n_classes) 
        
        else:
            self.aux1 = self.aux2 = None
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.maxpool(x)
        x=self.a3(x)
        x=self.b3(x)
        x=self.maxpool(x)
        x=self.a4(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x=self.b4(x)
        x=self.c4(x)
        x=self.d4(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x=self.e4(x)
        x=self.maxpool(x)
        x=self.a5(x)
        x=self.b5(x)
        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.dropout(x)
        x=self.fc(x)

        if self.aux_logits and self.training:
            return x, aux2, aux1
        return x 




### Test purpose
if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = InceptionNet(aux_logits=True, n_classes=1000)
    print("Done ! ")


