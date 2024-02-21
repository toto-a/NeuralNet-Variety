import torch
import os

debug = True
image_path = f"{os.getcwd()}/inference/data/Images"
captions_path = f"{os.getcwd()}/inference/data/captions.txt"
batch_size = 8
num_workers = 2
lr = 3e-3
lr_end=1e-3
warmup_epochs=1
weight_decay = 0.1
patience = 2
factor = 0.5
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embeddings = 2048
text_encoder_model = "distilbert-base-uncased"
text_embeddings = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = False # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
proj_dim = 256 
dropout = 0.1

