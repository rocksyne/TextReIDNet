"""
Doc.: Convert pytorch model to tensorRT
"""

import os
import sys

# 3rd party modules
from tqdm import tqdm as tqdm
import torch.nn as nn
import torch
from torch2trt import torch2trt

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from model.textreidnet import TextReIDNet
from config import sys_configuration

config:dict = sys_configuration(dataset_name="custom")
pretrained_model_path = os.path.join(config.model_save_path,"TextReIDNet_State_Dicts.pth.tar")
model = TextReIDNet(config)
model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
model = model.to(config.device)
model.eval() # put in inference mode


class TextEmbeddingModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, token_ids, ori_token_length):
        return self.original_model.text_embedding(token_ids, ori_token_length)

class ImageEmbeddingModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, image):
        return self.original_model.image_embedding(image)


# data samples
sample_image = torch.load('image_sample.pt').to(config.device)
sample_token_ids = torch.load('token_ids_sample.pt').to(config.device)
sample_ori_token_length = torch.load('orig_token_length_sample.pt').to(config.device)

# models
text_model = TextEmbeddingModel(model).eval().to(config.device)
image_model = ImageEmbeddingModel(model).eval().to(config.device)

# create trt models
text_model_trt = torch2trt(text_model, [sample_token_ids, sample_ori_token_length])
image_model_trt = torch2trt(image_model, [sample_image])


# now save the trt models
torch.save(text_model_trt.state_dict(), "text_embedding_trt_model.pth")
torch.save(image_model_trt.state_dict(), "image_embedding_trt_model.pth")