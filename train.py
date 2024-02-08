"""
C. January 2024
License: Please see LICENSE file
Doc.: Train TextReIDNet
"""

# System modules
import os
import warnings
import logging
import datetime

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
from evaluation.evaluations import calculate_similarity
from evaluation.evaluations import evaluate
from test_during_training import test, write_result


# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++[Global Configurations]++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.autograd.set_detect_anomaly(True)
#warnings.filterwarnings("ignore") 
config:dict = sys_configuration(dataset_name="CUHK-PEDES")
set_seed(config.seed) # using same as https://github.com/xx-adeline/MFPE/blob/main/src/train.py 
scaler = GradScaler() 

# Dataset stuff
train_data_loader, inference_img_loader, inference_txt_loader, train_num_classes = build_cuhkpedes_dataloader(config)

# Model and loss stuff
model = TextReIDNet(config).to(config.device)
identity_loss_fnx = IdentityLoss(config=config, class_num=train_num_classes).to(config.device)
ranking_loss_fnx = RankingLoss(config)

# Trainig configuration stuff
# Adapting https://github.com/xx-adeline/MFPE/blob/main/src/train.py#L36
cnn_params = list(map(id, model.visual_network.parameters()))
other_params = filter(lambda p: id(p) not in cnn_params, model.parameters())
other_params = list(other_params)
other_params.extend(list(identity_loss_fnx.parameters()))
param_groups = [{'params': other_params, 'lr': config.lr},
                {'params': model.visual_network.parameters(), 'lr': config.lr*0.1}]

optimizer = optim.AdamW(param_groups, betas=(config.adam_alpha, config.adam_beta))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.epoch_decay)
current_best_top1_accuracy:float = 0.

# Logging info
time_stamp = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
train_logger:logging = setup_logger(name='train_logger',log_file_path=config.train_log_path, write_mode=config.write_mode)
test_logger:logging  = setup_logger(name='test_logger',log_file_path=config.test_log_path, write_mode=config.write_mode)
train_logger.info("\n Started on {} \n {}".format(time_stamp,"="*35))
test_logger.info("\n Started on {} \n {}".format(time_stamp,"="*35))

if config.save_best_test_results_only:
    test_logger.info("\n Note: saving best test (inference) results only: \n")


if __name__ == '__main__':

    for current_epoch in range(1,config.epoch+1):

        # Conatainers for saving losses
        train_ranking_loss_list:list[float] = []
        train_identity_loss_list:list[float] = []
        train_total_loss_list:list[float] = []
        val_ranking_loss_list:list[float] = []
        val_identity_loss_list:list[float] = []
        val_total_loss_list:list[float] = []
 
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++[Train Model]+++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        # Use tqdm to see train progress
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
                precision_dtype = torch.bfloat16 if config.device== 'cpu' else torch.float16
                with torch.autocast(device_type=config.device,  dtype=precision_dtype):
                    visaual_embeddings, textual_embeddings = model(image, token_ids, orig_token_length)

                    # Calculate losses
                    train_ranking_loss = ranking_loss_fnx(visaual_embeddings, textual_embeddings, label)
                    train_identity_loss = identity_loss_fnx(visaual_embeddings, textual_embeddings, label)
                    train_total_loss = (config.ranking_loss_alpha*train_ranking_loss + config.identity_loss_beta*train_identity_loss)
                
                scaler.scale(train_total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # save losses
                train_ranking_loss_list.append(train_ranking_loss.item())
                train_identity_loss_list.append(train_identity_loss.item())
                train_total_loss_list.append(train_total_loss.item())

                # Prepare the progress bar display values
                values = {"Ranking Loss":np.mean(train_ranking_loss_list),
                          "Identity Loss":np.mean(train_identity_loss_list),
                          "Total Loss":np.mean(train_total_loss_list)}
                
                train_data_loader_progress.set_postfix(values) # update progress bar
        
        # write results for this ecpoch into log file
        txt_2_write = "Epoch: {} Ranking Loss: {:.5} Identity Loss: {:.5} Total Loss: {:.5}".format(current_epoch,
                                                                                                    np.mean(train_ranking_loss_list),
                                                                                                    np.mean(train_identity_loss_list),
                                                                                                    np.mean(train_total_loss_list))
        train_logger.info(txt_2_write)
           

        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++[Model Inference / Testing]++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        model.eval() # put in inference mode

        visaual_embeddings_list:list[torch.tensor] = []
        textual_embeddings_list:list[torch.tensor] = []
        visual_labels_list:list[torch.tensor] = []
        textual_labels_list:list[torch.tensor] = []

        # Extract images first
        for [image, label, _]  in tqdm(inference_img_loader,desc="Processing images.."):
            image = image.to(config.device)
            label = label.to(config.device)

            with torch.no_grad():
                visaual_embeddings = model.image_embedding(image)
            visaual_embeddings_list.append(visaual_embeddings)
            visual_labels_list.append(label.view(-1))
        
        # Now extract textual features
        for [label, token_ids, orig_token_length] in tqdm(inference_txt_loader,desc="Processing texts..."):
            label = label.to(config.device)
            token_ids = token_ids.to(config.device).long()
            orig_token_length = orig_token_length.to(config.device)

            with torch.no_grad():
                textual_embeddings = model.text_embedding(token_ids, orig_token_length)
            textual_embeddings_list.append(textual_embeddings)
            textual_labels_list.append(label.view(-1))
            
        visaual_embeddings_list = torch.cat(visaual_embeddings_list,0)
        textual_embeddings_list = torch.cat(textual_embeddings_list,0)
        visual_labels_list = torch.cat(visual_labels_list,0).cpu() # gallery / galleries depending on batch_size
        textual_labels_list = torch.cat(textual_labels_list,0).cpu() # queries depending on batch_size
       
        similarity = calculate_similarity(visaual_embeddings_list,textual_embeddings_list).numpy()
        ranks, mAP = evaluate(similarity.T, textual_labels_list, visual_labels_list) # calculate top ranks and mAP
        
        # logging / saving data
        txt_2_write = "Epoch: {} R1: {:.5}, R5: {:.5}, R10: {:.5}, map: {:.5}".format(current_epoch, ranks[0], ranks[4], ranks[9], mAP)
        if config.save_best_test_results_only and (ranks[0] > current_best_top1_accuracy):
            test_logger.info(txt_2_write)
            # save best model only

        else: # log everything, worse, same, or better
            test_logger.info(txt_2_write)
        
        print(txt_2_write) # show everything to console though
        # save the plots

        print("") # put space between each epoch progress bar


