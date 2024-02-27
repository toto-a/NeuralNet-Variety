import torch
import numpy as np
from tqdm import tqdm
import config as cfg
from clip_pretrained import CLIP
from utils import get_data,get_split
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt 
from PIL import Image



def get_image_embeddings(model):
    tokenizer= DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    valid_loader=get_data(get_split("valid"),tokenizer,"split")


    valid_image_embedding=[]
    with torch.no_grad():
        for batch in tqdm(valid_loader) :
            image_features=model.image_encoder(batch["image"])
            image_embeddin=model.image_projection(image_features)
            valid_image_embedding.append(image_embeddin)
    
    return torch.cat(valid_image_embedding)

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(cfg.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings=image_embeddings/image_embeddings.norm(dim=1,keepdim=True)
    text_embeddings=text_embeddings/text_embeddings.norm(dim=1,keepdim=True)
    
    dot_similarity = text_embeddings @ image_embeddings.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = Image.open(f"{cfg.image_path}/{match}")
        image=np.array(image)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()