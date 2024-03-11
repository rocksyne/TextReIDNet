"""
C. January 2024
License: Please see LICENSE file
Doc.: Person search using text query
"""

# System modules
import os, sys

# 3rd party modules
import torch
from tqdm import tqdm

# Application modules
sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from datasets.custom_dataloader import build_img_dataloader_from_dir, process_text_into_tokens
from model.textreidnet import TextReIDNet
from evaluation.evaluations import calculate_similarity
from PIL import Image


# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++[Global Configurations]++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')
#warnings.filterwarnings("ignore") 
config:dict = sys_configuration(dataset_name="custom")

# Dataset stuff
data_loader = build_img_dataloader_from_dir(config,os.path.join(config.project_parent_dir,'data','samples'))

# Model and loss stuff
pretrained_model_path = os.path.join(config.model_save_path,"TextReIDNet_State_Dicts.pth.tar")
model = TextReIDNet(config)
model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
model = model.to(config.device)
model.eval() # put in inference mode

# Description of the desired subject
textual_description = "The woman has long black hair. She is wearing blue pants, a white shirt and white shoes."


if __name__ == '__main__':

    print("")
    print('[INFO] Total parameters: {} million'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("")

    visaual_embeddings_list:list[torch.tensor] = []
    textual_embeddings_list:list[torch.tensor] = []
    image_names_list:list[str] = []

    print("") # just for printing aesthetics
    # Extract features from all images
    for image, image_name in tqdm(data_loader,desc="Processing images.."):
        image = image.to(config.device)

        with torch.no_grad():
            visaual_embeddings = model.image_embedding(image)
        visaual_embeddings_list.append(visaual_embeddings)
        image_names_list.extend(image_name)

    token_ids, orig_token_length = process_text_into_tokens(textual_description)
    token_ids = token_ids.unsqueeze(0).to(config.device).long()
    orig_token_length = torch.tensor([orig_token_length]).to(config.device)

    # Extract textual embeddings from query
    with torch.no_grad():
        textual_embeddings = model.text_embedding(token_ids, orig_token_length)
    textual_embeddings_list.append(textual_embeddings)

    visaual_embeddings_list = torch.cat(visaual_embeddings_list,0)
    textual_embeddings_list = torch.cat(textual_embeddings_list,0)

    similarity = calculate_similarity(visaual_embeddings_list,textual_embeddings_list).numpy()

    # Sort the indeces of similarity score, from the highest score to the lowest
    indices_sorted = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)
    
    # now sort the image list as well as the sim. scores
    ranked_image_list = [image_names_list[i] for i in indices_sorted]
    ranked_similarity_scores = [similarity[i] for i in indices_sorted]

    rank_1 = ranked_image_list[0]
    image = Image.open(rank_1 )
    image.save("retrieved_image.jpg")

    print("\n Retrieved (rank-1) image is `{}`.".format(rank_1))

    print("")
    print("All images ranked according to similarity score (descending order)")
    print("*"*70)
    for rank, [name, score] in enumerate(list(zip(ranked_image_list, ranked_similarity_scores)),start=1):
        print("{}. {}: {}".format(rank, name, score))



