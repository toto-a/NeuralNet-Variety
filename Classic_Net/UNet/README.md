

# U-Net

## Overview
In low dimension spaces, the network retain more spatial information than semantic in high dimension spaces. By using, skip connection, we are able to aggregate spatial and semantic information to the decoder, so the decoder doesn't only depend on the latent representation only.

## Structure
In my model as I used padding for the conv block, the input should be divisible by 2^(n_encoder_block) so 16 in my case. 


![unet](unet.png)
