

# An implementation from scratch of the Vision Transformer (ViT) from the paper _An image is worth 16*16 pixels_

## Positional Embeddings

Instead of using positional embeddings like mentionned in the paper, I decided to use learnable positional embeddings (see the reason [below](#Different-Positional-Encodings) )

Overall architecture : 

![vit](image/vision-transformer-vit.png)


## Different Positional Encodings






<p align="center">
  
  <img src=image/pe._sam.png>
  
  _Difference between sinusoidal embeddings in Transformers and fourier based encodings in recent model like SAM_
</p>

As we can see for the sinusoidal embeddings, if we take two groups of two pixels with the same distance between them. They will have differents encodings meaning they don't hold the same amount of information, which is somewhat wrong. The fourier based encodings resolves this issue



