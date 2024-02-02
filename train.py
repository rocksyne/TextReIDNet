"""
C. January 2024
License: Please see LICENSE file
Doc.: Train TextReIDNet
"""

# System modules
import os
import warnings

# 3rd party modules
import torch
import numpy as np
from torch import optim
from torch import autocast
from torch.cuda.amp import GradScaler

# Application modules
from config import sys_configuration
from utils.miscellaneous_utils import set_seed
from utils.miscellaneous_utils import  setup_logger
from datasets.cuhkpedes_dataloader import build_cuhkpedes_dataloader
from model.textreidnet import TextReIDNet

# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.autograd.set_detect_anomaly(True)
#warnings.filterwarnings("ignore") 

# Some global configurations
set_seed(3407) # using same as https://github.com/xx-adeline/MFPE/blob/main/src/train.py 
config = sys_configuration(dataset_name="CUHK-PEDES")
train_logger = setup_logger(name='train_logger',log_file_path=config.train_log_path)
test_logger  = setup_logger(name='test_logger',log_file_path=config.test_log_path)
model = TextReIDNet(config)

# dataset stuff
train_dataset, train_num_classes = build_cuhkpedes_dataloader(dataset_split='train', config=config)
val_dataset, val_num_classes = build_cuhkpedes_dataloader(dataset_split='val', config=config)
test_dataset, test_num_classes = build_cuhkpedes_dataloader(dataset_split='test', config=config)


print("Train: length:{} classes{} ".format(len(train_dataset), train_num_classes))
print("Val: length:{} classes{} ".format(len(val_dataset), val_num_classes))
print("Test: length:{} classes{} ".format(len(test_dataset), test_num_classes))



