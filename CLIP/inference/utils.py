import numpy as np
from torch.utils.data  import DataLoader
import config as cfg
import pandas as pd
from typing import Tuple
from clip_dataset import CLIPDataset

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
        dataset_split["image"].values,
        dataset_split["caption"].values,
        tokenizer,
        split,

    )

    dataloader=DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True if split=="train" else False,
        num_workers=cfg.num_workers,
    )
    

    return dataloader



class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def update_lr(optimizer, lr_schedule,step):
    for param_group in optimizer.param_groups:
         param_group["lr"]=lr_schedule[step]


def get_num_batches(dataloader : DataLoader) :
    num_samples=len(dataloader)
    num_batches=num_samples//dataloader.batch_size

    return num_batches

def cosine_scheduler(init_value, final_value, epochs, iter_per_epoch, warmup, warmup_start_value =0):

    warmup_schedule=[]
    if warmup>0 :
        warmup_schedule=np.arange(warmup_start_value, init_value )
    


    iters=np.arange(epochs * iter_per_epoch - warmup)
    cosine_lr=final_value + 0.5*(init_value-final_value) *(1+np.cos(np.pi*iters/len(iters)))

    schedule=np.concatenate((warmup_schedule,cosine_lr))
    assert len(schedule)==epochs*iter_per_epoch

    return schedule

    

