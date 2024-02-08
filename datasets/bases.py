"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main
"""

import torch
from torch.utils.data import Dataset
from utils.iotools import read_image
from utils.miscellaneous_utils import pad_tokens
from datasets.tiktoken_tokenizer import TikTokenizer
from datasets.simple_tokenizer import SimpleTokenizer
from datasets.bert_tokenizer import BERTTokenizer
import numpy as np


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 tokens_length_max: int = 100, 
                 tokenizer_type:str="bert"):
        
        self.dataset = dataset
        self.transform = transform
        self.tokens_length_max = tokens_length_max
        self.tokenizer_type = tokenizer_type

        # create the appropriate tokenizer type
        if tokenizer_type == "bert":
            self.tokenizer = BERTTokenizer()

        elif tokenizer_type == "simple_tokenizer":
            self.tokenizer = SimpleTokenizer()
        
        elif tokenizer_type == "tiktoken_cl100k":
            self.tokenizer = TikTokenizer(encoding_base='cl100k_base')

        elif tokenizer_type == "tiktoken_p50k":
            self.tokenizer = TikTokenizer(encoding_base='p50k_base')
        
        elif tokenizer_type == "tiktoken_r50k":
            self.tokenizer = TikTokenizer(encoding_base='r50k_base')
        
        else:
            raise NotImplemented("No implemetation for `{}` tokenization type")
            

        
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        pid = torch.from_numpy(np.array([pid])).long()

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = self.tokenizer(caption) # eg. torch.tensor([1165, 13, 564, 74, ..., 1167])
        token_ids, orig_token_length  = pad_tokens(tokens, self.tokens_length_max)

        ret = {'pids': pid,
               'image_ids': image_id,
               'img_path': img_path,
               'images': img,
               'token_ids': token_ids.to(torch.long),
               'orig_token_length': orig_token_length,
               'caption':caption
               }

        return ret


    


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, img_path


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 tokenizer_type:str="bert",
                 tokens_length_max: int = 100):
        
        self.caption_pids = caption_pids
        self.captions = captions
        self.tokenizer_type = tokenizer_type
        self.tokens_length_max = tokens_length_max

        # create the appropriate tokenizer type
        if tokenizer_type == "bert":
            self.tokenizer = BERTTokenizer()

        elif tokenizer_type == "simple_tokenizer":
            self.tokenizer = SimpleTokenizer()
        
        elif tokenizer_type == "tiktoken_cl100k":
            self.tokenizer = TikTokenizer(encoding_base='cl100k_base')

        elif tokenizer_type == "tiktoken_p50k":
            self.tokenizer = TikTokenizer(encoding_base='p50k_base')
        
        elif tokenizer_type == "tiktoken_r50k":
            self.tokenizer = TikTokenizer(encoding_base='r50k_base')
        
        else:
            raise NotImplemented("No implemetation for `{}` tokenization type")


    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        label, caption = self.caption_pids[index], self.captions[index]
        tokens = self.tokenizer(caption)
        token_ids, orig_token_length = pad_tokens(tokens, self.tokens_length_max)

        return label, token_ids, orig_token_length