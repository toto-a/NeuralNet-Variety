import os
import torch
from PIL import Image
from typing import Tuple
from torch.utils.data  import Dataset, DataLoader
import albumentations as A
import albumentations.pytorch.transforms as T
import numpy as np

import config as cfg


class CLIPDataset(Dataset):
    def __init__(self, images, captions , tokenizer ,transform: True, mode :str="train"):
        self.transform = transform
        self.mode=mode
        self.images_path = images
        self.captions =list(captions)

        self.encoded_captions=tokenizer(
            self.captions,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        )
        
    

    @staticmethod
    def _get_transform(mode) : 
        if mode == "train":
            return A.Compose([
                A.Resize(cfg.size, cfg.size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return A.Compose([
                A.Resize(cfg.size, cfg.size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:

        item={
            key : torch.tensor(values[idx])
            for key,values in self.encoded_captions.items()
        }
        image = self.images_path[idx]
        caption = self.captions[idx]
        image = Image.open(os.path.join(cfg.image_path,image))
        if self.transform:
            t = self._get_transform(self.mode)
            image=t(image=np.array(image))
        

        item["image"]=torch.tensor(image["image"]).permute(2,1,0).float()
        item["caption"]=caption
        return item