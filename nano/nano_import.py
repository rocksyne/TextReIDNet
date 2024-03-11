# -*- coding: utf-8 -*-
import torch
import os
import sys
import time

# 3rd party modules
from tqdm import tqdm as tqdm
from torch2trt import TRTModule
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Application modules
# First add project parent directory to sys.path
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,PROJ_DIR) 

from model.model import TextImgPersonReidNet
from utils.configuration import sys_configuration

# Some global configurations
opt = sys_configuration()
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


do_image = TRTModule()
do_text  = TRTModule()

do_image.load_state_dict(torch.load('image_embedding.pth'))
do_text.load_state_dict(torch.load('text_embedding.pth'))

# Text embedding model
sample_caption_code = torch.load("caption_code.pt").to(opt.device) # Example input
sample_caption_length = torch.load("caption_length.pt").to(opt.device)
sample_image = torch.randn(1, 3, 348, 128, device=opt.device)  # Example image input

for ii in range(50):

    print("")
    print("Iteration: ",ii)
    start_time = time.time()
    text_global_i, text_local_i = do_text(sample_caption_code, sample_caption_length)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Single text process {duration} seconds.")


    start_time = time.time()
    img_global_i, img_local_i = do_image(sample_image)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Single image process {duration} seconds.")
        
