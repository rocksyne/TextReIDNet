"""
C. January 2024
License: Please see LICENSE file
Doc.: Test TextReIDNet
"""
import os

# 3rd party modules
import torch
from tqdm import tqdm

# Application modules
from config import sys_configuration
from datasets.cuhkpedes_dataloader import build_cuhkpedes_dataloader
from model.textreidnet import TextReIDNet
from evaluation.evaluations import calculate_similarity
from evaluation.evaluations import evaluate


# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++[Global Configurations]++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.autograd.set_detect_anomaly(True)
#warnings.filterwarnings("ignore") 
config:dict = sys_configuration(dataset_name="CUHK-PEDES")

# Dataset stuff
train_data_loader, inference_img_loader, inference_txt_loader, train_num_classes = build_cuhkpedes_dataloader(config)

# Model and loss stuff
pretrained_model_path = os.path.join(config.model_save_path,"TextReIDNet_State_Dicts.pth.tar")
model = TextReIDNet(config)
model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
model = model.to(config.device)


if __name__ == '__main__':

    print("")
    print('[INFO] Total parameters: {} million'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("")

    model.eval() # put in inference mode

    visaual_embeddings_list:list[torch.tensor] = []
    textual_embeddings_list:list[torch.tensor] = []
    visual_labels_list:list[torch.tensor] = []
    textual_labels_list:list[torch.tensor] = []

    print("") # just for printing aesthetics
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
    textual_labels_list = torch.cat(textual_labels_list,0).cpu() # query / queries depending on batch_size
    
    similarity = calculate_similarity(visaual_embeddings_list,textual_embeddings_list).numpy()
    ranks, mAP = evaluate(similarity.T, textual_labels_list, visual_labels_list) # calculate top ranks and mAP
    
    # logging / saving data
    txt_2_write = "Top_1: {:.4}, Top_5: {:.4}, Top_10: {:.4}, mAP: {:.4}".format(ranks[0], ranks[4], ranks[9], mAP)
    print(txt_2_write) # show everything to console though
    print("") # just for printing aesthetics




