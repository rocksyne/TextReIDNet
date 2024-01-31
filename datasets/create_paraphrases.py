# -*- coding: utf-8 -*-
# System modules
import os
import sys
import json


# 3rd party modules
import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartTokenizer

# Application modules
# First add project parent directory to sys.path
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,PROJ_DIR) 

# Application modules
from datasets import build_dataloader
from utils.configuration import sys_configuration
from tqdm import tqdm as tqdm


# Some global configurations
opt = sys_configuration()

# See https://pytorch.org/docs/stable/multiprocessing.html
torch.multiprocessing.set_sharing_strategy('file_system')


class Paraphrase(nn.Module):

    def __init__(self, model_name="eugenesiow/bart-paraphrase"):
        """
        model_name: `facebook/bart-large-cnn`, or `eugenesiow/bart-paraphrase`
        """
        super(Paraphrase, self).__init__()

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(opt.device)
        self.max_lenght = 150 # more then enough since we are truncating ours to 100
    
    def forward(self,input_sentences:list[str]=None)-> str:

        if isinstance(input_sentences,list) is False:
            raise ValueError("input_sentences must be a list. Current type is a {}".format(type(input_sentences)))
        
        inputs = self.tokenizer([sentence for sentence in input_sentences], 
                       return_tensors="pt", max_length=self.max_lenght, truncation=True, padding=True)
        
        paraphrase_ids = self.model.generate(inputs["input_ids"].to(opt.device), attention_mask=inputs["attention_mask"].to(opt.device), 
                                    max_length=self.max_lenght, num_beams=4, early_stopping=True)
        
        paraphrases = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                   for g in paraphrase_ids]

        return paraphrases



paraphrase_sentence:Paraphrase = Paraphrase()
paraphased_dict:dict[int,str] = {}

train_loader, _, _, _ = build_dataloader(dataset_name="CUHKPEDES", tranforms=None, mode='training')

for train_data in tqdm(train_loader):

    captions  = train_data['caption']
    person_id = train_data['pids']

    pid_caption_paraphrase = zip(person_id,captions,paraphrase_sentence(captions))

    for curr_pid, curr_caption, curr_paraphrase in pid_caption_paraphrase:
        paraphased_dict[curr_pid.item()] = curr_paraphrase

with open('./paraphrased_captions.json', 'w+') as file:
    json.dump(paraphased_dict, file)

print("[Info] Paraphase file generation completed.")
    

           
