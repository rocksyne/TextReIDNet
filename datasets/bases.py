from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.tiktoken_tokenizer import TikTokenizer
from utils.simple_tokenizer import SimpleTokenizer
from utils.bert_tokenizer import BERTTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import numpy as np
import transformers as ppb



def get_caption_mask(caption, caption_length_max):
    """
    This entire code snippet is adapted from the URL below.
    https://github.com/xx-adeline/MFPE/blob/main/src/data/dataset.py#L65
    Enhancement: maske tensors contigeous for memory efficiency
    """
    caption_length = len(caption)
    caption = torch.from_numpy(np.array(caption)).view(-1).contiguous().float() # make contiguous
    if caption_length < caption_length_max:
        zero_padding = torch.zeros(caption_length_max - caption_length)
        caption = torch.cat([caption, zero_padding], 0)
    else:
        caption = caption[:caption_length_max]
        caption_length = caption_length_max
    caption_mask = np.where(caption != 0, 1, 0)
    caption_mask = torch.from_numpy(caption_mask)
    return caption, caption_length, caption_mask


def tokenize(caption: str, tokenizer, text_length=100, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(f"Input {caption} is too long for context length {text_length}")
    result[:len(tokens)] = torch.tensor(tokens)
    return result



class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))





class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 caption_length_max: int = 100,
                 truncate: bool = True,
                 tokenizer_type:str="bert"):
        
        self.dataset = dataset
        self.transform = transform
        self.caption_length_max = caption_length_max
        self.truncate = truncate
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
            
        print("Tokenizer: ",tokenizer_type)

        

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]

        # Converting pid into a list[int] approach is inspired and adopted from 
        # https://github.com/xx-adeline/MFPE/blob/main/src/data/dataset.py#L59
        pid = torch.from_numpy(np.array([pid])).long()

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = self.tokenizer(caption)
        caption_code, caption_length, caption_mask = get_caption_mask(tokens, self.caption_length_max)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_codes': caption_code.to(torch.long),
            'caption_length': caption_length,
            'caption_mask': caption_mask,
            'img_path': img_path
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
                 caption_length_max: int = 100,
                 truncate: bool = True):
        
        self.caption_pids = caption_pids
        self.captions = captions
        self.tokenizer_type = tokenizer_type
        self.caption_length_max = caption_length_max
        self.truncate = truncate

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
            
        print("Tokenizer: ",tokenizer_type)


    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        label, caption = self.caption_pids[index], self.captions[index]
        tokens = self.tokenizer(caption)
        caption_code, caption_length, caption_mask = get_caption_mask(tokens, self.caption_length_max)

        return label, caption_code, caption_length, caption_mask