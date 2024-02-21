import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import get_lr, AvgMeter, get_data, make_data, get_split,cosine_scheduler
from tqdm import tqdm
from transformers import DistilBertTokenizer
import config as cfg
from clip_pretrained import CLIP

@torch.no_grad()
def estimate_loss(model,valid_loader=get_split("valid")):
    
    loss=AvgMeter()
    loss_meter =AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    model.eval()
    for i,batch in enumerate(tqdm_object):
        image,caption=batch["image"], batch["caption"]
        count=batch["image"].size(0)
        loss_meter.update(loss.item(),count)

        tqdm_object.set_postfix(loss_meter.avg())
    
    model.train()
    return loss_meter



def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter =AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for i,batch in enumerate(tqdm_object):

        image,caption=batch["image"],batch["caption"]
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



def main() :

    ## Getting the dataset
    split_df=get_split()
    datal=get_data(split_df,tokenizer=DistilBertTokenizer.from_pretrained(cfg.text_tokenizer), split="train")
    
    model=CLIP(temperature=cfg.temperature, image_embeddings=cfg.image_embeddings, text_embeddings=cfg.text_embeddings)
    model=model.to(cfg.device)

    optimizer=optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)

    best_acc=float('inf')
    for ep in range(cfg.epochs):

        model.train()
        loss=train_epoch(model,datal,optimizer,cosine_scheduler())
        valid_loss=estimate_loss(model)

        is_best=valid_loss<best_acc
        if is_best :
            best_acc=valid_loss
            best_loss=best_acc

        torch.save(
            {
                "epoch ": ep + 1 ,
                "state_dict" : model.state_dict,
                "optimizer " : optimizer.state_dict,
                "best_loss ": best_loss
            },
            "./state/checkpoint.pt"
        )