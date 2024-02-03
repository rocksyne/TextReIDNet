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
from tqdm import tqdm
from torch import optim
from torch import autocast
from torch.cuda.amp import GradScaler


# Application modules
from config import sys_configuration
from utils.miscellaneous_utils import set_seed
from utils.miscellaneous_utils import  setup_logger
from datasets.cuhkpedes_dataloader import build_cuhkpedes_dataloader
from model.textreidnet import TextReIDNet
from evaluation.ranking_loss import RankingLoss
from evaluation.identity_loss import IdentityLoss


# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.autograd.set_detect_anomaly(True)
#warnings.filterwarnings("ignore") 

# Some global configurations
config:dict = sys_configuration(dataset_name="CUHK-PEDES")
train_logger:logging = setup_logger(name='train_logger',log_file_path=config.train_log_path)
test_logger:logging  = setup_logger(name='test_logger',log_file_path=config.test_log_path)
set_seed(config.seed) # using same as https://github.com/xx-adeline/MFPE/blob/main/src/train.py 
scaler = GradScaler() 

# Dataset stuff
dataset_loader:dict[str,torch.tensor] = build_cuhkpedes_dataloader(config)
train_data_loader = dataset_loader['train_data_loader']
train_num_classes = dataset_loader['train_num_classes']
val_data_loader = dataset_loader['val_data_loader']
val_num_classes = dataset_loader['val_num_classes']
test_data_loader = dataset_loader['test_data_loader']
test_num_classes = dataset_loader['test_num_classes']

# Model and loss stuff
# We have two identity loss instances because of the difference in the
# classes between the train and the validatin sets
model = TextReIDNet(config).to(config.device)
train_identity_loss_fnx = IdentityLoss(config=config, class_num=train_num_classes).to(config.device)
val_identity_loss_fnx = IdentityLoss(config=config, class_num=val_num_classes).to(config.device)
ranking_loss_fnx = RankingLoss(config)

# Trainig configuration stuff
# Adapting https://github.com/xx-adeline/MFPE/blob/main/src/train.py#L36
cnn_params = list(map(id, model.visual_network.parameters()))
other_params = filter(lambda p: id(p) not in cnn_params, model.parameters())
other_params = list(other_params)
other_params.extend(list(train_identity_loss_fnx.parameters()))
param_groups = [{'params': other_params, 'lr': config.lr},
                {'params': model.visual_network.parameters(), 'lr': config.lr*0.1}]

optimizer = optim.AdamW(param_groups, betas=(config.adam_alpha, config.adam_beta))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.epoch_decay)


if __name__ == '__main__':

    for current_epoch in range(1,config.epoch+1):

        # Conatainers for saving losses
        train_ranking_loss_list:list[float] = []
        train_identity_loss_list:list[float] = []
        train_total_loss_list:list[float] = []
        val_ranking_loss_list:list[float] = []
        val_identity_loss_list:list[float] = []
        val_total_loss_list:list[float] = []

        # Train first. Use tqdm to see train progress
        # https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e
        model.train() # ut in train mode
        with tqdm(train_data_loader, unit='batch') as train_data_loader_progress:
            train_data_loader_progress.set_description(f"Train - Epoch {current_epoch} of {config.epoch}")

            for train_data_batch in train_data_loader_progress:
                image = train_data_batch['images'].to(config.device)
                label = train_data_batch['pids'].to(config.device)
                token_ids = train_data_batch['token_ids'].to(config.device)
                orig_token_length = train_data_batch['orig_token_length'].to(config.device)

                # Zero-grad before making prediction with model
                # https://pytorch.org/docs/stable/optim.html
                optimizer.zero_grad()

                # Use mixed precision training
                with torch.autocast(device_type=config.device,  dtype=torch.float16):
                    visaual_embeddings, textual_embeddings = model(image, token_ids, orig_token_length)

                    # Calculate losses
                    train_ranking_loss = ranking_loss_fnx(visaual_embeddings, textual_embeddings, label)
                    train_identity_loss = train_identity_loss_fnx(visaual_embeddings, textual_embeddings, label)
                    train_total_loss = (config.ranking_loss_alpha*train_ranking_loss + config.identity_loss_beta*train_identity_loss)
                
                scaler.scale(train_total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # save losses
                train_ranking_loss_list.append(train_ranking_loss.item())
                train_identity_loss_list.append(train_identity_loss.item())
                train_total_loss_list.append(train_total_loss.item())

                # Prepare the progress bar display values
                values = {"T. Ranking Loss":np.mean(train_ranking_loss_list),
                          "T. Identity Loss":np.mean(train_identity_loss_list),
                          "T. Total Loss":np.mean(train_total_loss_list)}
                
                train_data_loader_progress.set_postfix(values) # update progress bar
           

        # Now do val. Use tqdm to see validation progress
        model.eval() # put in inference mode
        with tqdm(val_data_loader, unit='batch') as val_data_loader_progress:
            val_data_loader_progress.set_description(f"Train - Epoch {current_epoch} of {config.epoch}")

            for val_data_batch in val_data_loader_progress:
                image = val_data_batch['images'].to(config.device)
                label = val_data_batch['pids'].to(config.device)
                token_ids = val_data_batch['token_ids'].to(config.device)
                orig_token_length = train_data_batch['orig_token_length'].to(config.device)

                # no gradient related operations
                with torch.no_grad():
                    visaual_embeddings, textual_embeddings = model(image, token_ids, orig_token_length)

                # Calculate losses
                val_ranking_loss = ranking_loss_fnx(visaual_embeddings, textual_embeddings, label)
                val_identity_loss = val_identity_loss_fnx(visaual_embeddings, textual_embeddings, label)
                val_total_loss = (config.ranking_loss_alpha*val_ranking_loss + config.identity_loss_beta*val_identity_loss)
                
                # save losses
                val_ranking_loss_list.append(val_ranking_loss.item())
                val_identity_loss_list.append(val_identity_loss.item())
                val_total_loss_list.append(val_total_loss.item())

                # Prepare the progress bar display values
                values = {"T. Ranking Loss":np.mean(val_ranking_loss_list),
                          "T. Identity Loss":np.mean(val_identity_loss_list),
                          "T. Total Loss":np.mean(val_total_loss_list)}
                
                val_data_loader_progress.set_postfix(values) # update progress bar
                
        print("") # put space between each epoch progress bar


