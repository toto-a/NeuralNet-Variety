import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import get_lr, AvgMeter, get_data, make_data, get_split,cosine_scheduler, update_lr, get_num_batches, save_ckp
from tqdm import tqdm
from transformers import DistilBertTokenizer
import config as cfg
import os
from clip_pretrained import CLIP

@torch.no_grad()
def estimate_loss(model,valid_df=get_split("valid")):
    tokenizer=DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    valid_loader=get_data(valid_df,tokenizer=tokenizer,split="valid")
    
    loss_meter =AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    model.eval()
    for i,batch in enumerate(tqdm_object):
        image,caption=batch["image"], batch["caption"]


        loss=model(batch)
        count=batch["image"].size(0)
        loss_meter.update(loss.item(),count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    
    model.train()
    return loss_meter



def train_epoch(model, train_loader, optimizer, lr_scheduler, epoch):
    loss_meter =AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    num_batches=get_num_batches(train_loader)
    for i,batch in enumerate(tqdm_object):

        step=num_batches*epoch + i 
        update_lr(optimizer,lr_scheduler,step)

        image,caption=batch["image"],batch["caption"]

        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

        ## Image
        count = image.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter




def main() :

    ## Getting the dataset
    split_df=get_split()
    datal=get_data(split_df,tokenizer=DistilBertTokenizer.from_pretrained(cfg.text_tokenizer), split="train")
    
    model=CLIP(temperature=cfg.temperature, image_embeddings=cfg.image_embeddings, text_embeddings=cfg.text_embeddings)
    model=model.to(cfg.device)

    optimizer=optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    cs_lr=cosine_scheduler(cfg.lr,cfg.lr_end,cfg.epochs,get_num_batches(datal),cfg.warmup_epochs,cfg.warmuup_start_value)


    best_acc=float('inf')
    for ep in range(cfg.epochs):

        model.train()
        loss=train_epoch(model,datal,optimizer,cs_lr,ep)
        valid_loss=estimate_loss(model)

        is_best=valid_loss.avg<best_acc
        if is_best :
            best_acc=valid_loss
            best_loss=best_acc

        if not os.path.exists("./inference/state/") : 
            print("Exists ! ")
            os.makedirs("./inference/state/")


        checkpoint={
                "epoch": ep + 1 ,
                "state_dict" : model.state_dict(),
                "optimizer " : optimizer.state_dict(),
                "best_loss": best_loss,
                "valid_loss" : valid_loss
            },
        save_ckp(checkpoint, is_best,"./inference/state/")



if __name__=="__main__":

    main()
