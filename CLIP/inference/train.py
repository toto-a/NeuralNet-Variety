import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import get_lr, AvgMeter
from tqdm import tqdm
from transformers import DistilBertTokenizer
import pandas as pd
import numpy as np

import config as cfg
from typing import Tuple
from clip_dataset import CLIPDataset
from clip_pretrained import CLIP

def make_data(captions_path : str =cfg.captions_path) :
    captions_df=pd.read_csv(captions_path)
    max_ids=captions_df.shape[0] +1
    image_ids=np.arange(0,max_ids) 
    captions_df["id"]=image_ids[:-1]

    ## Set the see for reproducibility
    np.random.seed(42)
    valids_ids=np.random.choice(image_ids, size=int(0.2*max_ids), replace=False) 
    train_ids=np.setdiff1d(image_ids,valids_ids)

    train_df=captions_df[captions_df["id"].isin(train_ids)]
    valid_df=captions_df[captions_df["id"].isin(valids_ids)]

    return train_df,valid_df

    
def get_split(split :str="train"):
    train_df , valid_df=make_data()
    return train_df if split=="train" else valid_df
    

def get_data(dataset_split,tokenizer, split):
    dataset=CLIPDataset(
        dataset_split["images"].values,
        dataset_split["captions"].values,
        tokenizer,
        split,

    )

    dataloader=DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    return dataloader



def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter =AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        # batch = {k: v.to(cfg.device) for k, v in batch.items() if k != "caption"}
        image,caption, attention_mask=batch
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        ## Image
        count = image.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def get_image_embeddings(model, image_loader):
    tokenizer= DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    valid_loader=get_data("valid")

    model=CLIP().to(cfg.device)
    model.eval()

    valid_image_embedding=[]
    with torch.no_grad():
        for batch in tqdm(valid_loader) :
            image_features=model.image_encoder(batch["image"])
            image_embeddin=model.image_projection(image_features)
            valid_image_embedding.append(image_embeddin)
    
    return model, torch.cat(valid_image_embedding)


    return model.image_encoder(image_loader,tokenizer)  


if __name__=="__main__" :

    split_df=get_split()
    datal=get_data(split_df,tokenizer=DistilBertTokenizer, split="train")