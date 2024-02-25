import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import config as cfg

from clip_part import ImageEncoder, TextEncoder, ProjectionHead



class CLIP(nn.Module):
    def __init__(self, temperature : cfg.temperature, image_embeddings : cfg.image_embeddings, text_embeddings : cfg.text_embeddings):
        super(CLIP, self).__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(image_embeddings)
        self.text_projection = ProjectionHead(text_embeddings)
        self.temperature=temperature
    

    def forward(self, batch : Tuple[torch.Tensor , str, torch.Tensor]) :

        image,input_ids, attention_mask=batch["image"], batch["input_ids"], batch["attention_mask"]
        image_encoded=self.image_encoder(image)
        caption_features=self.text_encoder(input_ids,
                                            attention_mask)
        
        ### Getting the embeddings (image and text)
        image_embeddings=self.image_projection(image_encoded)
        text_embeddings=self.text_projection(caption_features)

        image_embeddings=image_embeddings/image_embeddings.norm(dim=1,keepdim=True)
        text_embeddings=text_embeddings/text_embeddings.norm(dim=1,keepdim=True)
        ## Loss

        image_embeddings=image_embeddings / self.temperature
        logits_per_text= text_embeddings @ image_embeddings.T 
        logits_per_image= image_embeddings @ text_embeddings.T

        targets =  F.softmax((logits_per_image + logits_per_text)/2 *self.temperature, dim=-1)

        texts_loss = F.cross_entropy(logits_per_text, targets, reduction='none')
        images_loss = F.cross_entropy(logits_per_image, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 
        return loss.mean()




