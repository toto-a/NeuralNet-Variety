import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F




class Params :
    num_hiddens :int  = 128
    num_residual_layers : int = 2 
    num_residual_hiddens : int = 32
    num_embeddings :int = 512
    embedding_dim : int = 64
    commitment_cost : int = 0.25








class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
    
    
    def forward(self, inputs):
        
        #BCHW -> BHWC
        inputs=inputs.permute(0,2,3,1).contiguous()
        input_shape=inputs.shape
        flat_inputs=inputs.view(-1,self._embedding_dim)

        distances=(
            torch.sum(flat_inputs**2,dim=1,keepdim=True) +
            torch.sum(self._embedding.weight**2,dim=1) -
            2*torch.matmul(flat_inputs,self._embedding.weight.T)

        )

        ## Get the nearest embeddings and select embd weight
        encodings_indices=torch.argmin(distances, dim=1, keepdim=True)
        encodings=F.one_hot(encodings_indices,num_classes=self._num_embeddings).float()
        quantized_x=torch.matmul(encodings,self._embedding.weight).view(input_shape)


        ### Losses
        
        codebook_loss=F.mse_loss(quantized_x.detach(), inputs)
        commitment_loss=F.mse_loss(quantized_x, inputs.detach())
        loss=codebook_loss + self._commitment_cost*commitment_loss

        quantized_x=inputs - (quantized_x-inputs).detach()
        avg_probs=torch.mean(encodings,0)
        perplexity=torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))


        return loss,quantized_x.permute(0,3,1,2).contiguous(), perplexity, encodings




class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

        
        

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)






class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
        



class Model(nn.Module):
    def __init__(self, dims : Params) -> None:
        super().__init__() 
        self.dims=dims

        self._encoder = Encoder(3, dims.num_hiddens,
                                dims.num_residual_layers, 
                                dims.num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=dims.num_hiddens, 
                                      out_channels=dims.embedding_dim,
                                      kernel_size=1, 
                                      stride=1)  
        

        self._vq_vae = VectorQuantizer(dims.num_embeddings, dims.embedding_dim,
                                           dims.commitment_cost)
        self._decoder = Decoder(dims.embedding_dim,
                                dims.num_hiddens, 
                                dims.num_residual_layers, 
                                dims.num_residual_hiddens)
        

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity