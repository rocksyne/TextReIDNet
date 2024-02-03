"""
C. January 2024
License: Please see LICENSE file
Doc.: Train TextReIDNet
"""

# System modules
import os
import warnings
import logging

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
config:dict = sys_configuration(dataset_name="CUHK-PEDES")
train_logger:logging = setup_logger(name='train_logger',log_file_path=config.train_log_path)
test_logger:logging  = setup_logger(name='test_logger',log_file_path=config.test_log_path)
model = TextReIDNet(config)

# Dataset stuff
dataset_loader:dict[str,torch.tensor] = build_cuhkpedes_dataloader(config)
train_data_loader = dataset_loader['train_data_loader']
train_num_classes = dataset_loader['train_num_classes']
val_data_loader = dataset_loader['val_data_loader']
val_num_classes = dataset_loader['val_num_classes']
test_data_loader = dataset_loader['test_data_loader']
test_num_classes = dataset_loader['test_num_classes']

# Trainig parameters
cnn_params = list(map(id, model.efficient_net.parameters()))
other_params = filter(lambda p: id(p) not in cnn_params, model.parameters())
other_params = list(other_params)
other_params.extend(list(id_loss_fun_global.parameters()))
other_params.extend(list(id_loss_fun_local.parameters()))
param_groups = [{'params': other_params, 'lr': opt.lr},
                {'params': model.efficient_net.parameters(), 'lr': config.lr * 0.1}]




