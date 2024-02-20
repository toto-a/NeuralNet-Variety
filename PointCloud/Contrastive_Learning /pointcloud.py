

import torch
version = f"https://data.pyg.org/whl/torch-{torch.__version__}.html"
try:
    import torch_geometric
except:
    # !echo $version
    # !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f $version
    import torch_geometric

### ShapeNet dataset
from torch_geometric.datasets import ShapeNet
dataset=ShapeNet(root="",categories=["Table","Lamp","Guitar","Motorbike"]).shuffle()[:4000]

import numpy as np
import plotly.express  as px

def plot_3d_shape(shape) :
  x=shape.pos[:,0]
  y=shape.pos[:,1]
  z=shape.pos[:,2]

  fig=px.scatter_3d(x=x,y=y,z=z,opacity=0.3)
  fig.show()


ix=np.arange(0,4000)
idx=np.random.choice(ix)
plot_3d_shape(dataset[idx])

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

## Data Augmentation

data_loader= DataLoader(dataset, batch_size=32, shuffle=True)
augmentation=T.Compose([T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)])

## Sample again from the loader

sample=next(iter(data_loader))
print(sample)
plot_3d_shape(sample[20])

### Transformed datapoint

transformed=augmentation(sample)
plot_3d_shape(transformed[25])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

class Model(nn.Module):
  def __init__(self, k=20, aggr="max") :
    # kNN neighbor, aggr-> aggregation strategy (global max pool here)
    super().__init__()

    ##Features Extraction (Encoder)

    self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)
    self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
    # Encoder head
    self.lm_head = nn.Linear(128 + 64, 128)
    # Projection head (See explanation in SimCLRv2)
    self.mlp = MLP([128, 256, 32], norm=None)

  def forward(self, data, train="True") :
    if train :
      augm_1=augmentation(data)
      augm_2=augmentation(data)

      pos_1, batch_1=augm_1.pos, augm_1.batch
      pos_2, batch_2=augm_2.pos, augm_2.batch

      f1=self.conv1(pos_1, batch_1)
      f2=self.conv2(f1, batch_1)
      print(self.lm_head)
      hp1=self.lm_head(torch.cat([f1,f2],dim=1))


      f1=self.conv1(pos_2, batch_2)
      f2=self.conv2(f1, batch_2)
      print(torch.cat([f1,f2],dim=1).size())
      hp2=self.lm_head(torch.cat([f1,f2],dim=1))
      print("2 : ",self.lm_head)

      # Global representations
      h1=global_max_pool(hp1,batch_1)
      h2=global_max_pool(hp2,batch_2)

    else :
        f1=self.conv1(data.pos, data.batch)
        f2=self.conv2(f1, data.batch)
        h=self.lm_head(torch.cat([f1,f2],dim=1))
        return global_max_pool(h,data.batch)

    compact_h_1=self.mlp(h1)
    compact_h_2=self.mlp(h2)


    return h1, h2, compact_h_1, compact_h_2

# !pip install pytorch-metric-learning

from pytorch_metric_learning.losses import NTXentLoss
loss_func= NTXentLoss(temperature=0.10)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.optim as optim
cmodel=Model().to(device)
optimizer=optim.Adam(cmodel.parameters(), lr= 0.001)
scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

dataloader=DataLoader(dataset, batch_size=32, shuffle=True)

import tqdm


def train():
    cmodel.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        # Get data representations
        h_1, h_2, compact_h_1, compact_h_2 = cmodel(data)
        # Prepare for loss
        embeddings = torch.cat((compact_h_1, compact_h_2))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataset)

for epoch in range(1, 4):
    loss = train()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    scheduler.step()

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Get sample batch
sample = next(iter(data_loader))

# Get representations
h = cmodel(sample.to(device), train=False)
h = h.cpu().detach()
labels = sample.category.cpu().detach().numpy()

# Get low-dimensional t-SNE Embeddings
h_embedded = TSNE(n_components=2, learning_rate='auto',
                   init='random').fit_transform(h.numpy())

# Plot
ax = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1], hue=labels,
                    alpha=0.5, palette="tab10")

# Add labels to be able to identify the data points
annotations = list(range(len(h_embedded[:,0])))

def label_points(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(int(point['val'])))


label_points(pd.Series(h_embedded[:,0]),
            pd.Series(h_embedded[:,1]),
            pd.Series(annotations),
            plt.gca())

# ## Code to free cuda space in colab

# import gc
# # del model
# del cmodel
# gc.collect()
# torch.cuda.empty_cache()

