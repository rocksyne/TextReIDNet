"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main
"""

# System modules
import os.path as op
from typing import List

# 3rd party modules
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Application modules
from utils.iotools import read_json
from utils.miscellaneous_utils import collate
from datasets.cuhkpedes import CUHKPEDES
from datasets.bases import ImageTextDataset



def get_transform(dataset_split:str=None, config:dict=None):
    """
    Doc.:   Get the appropriate transform
    Args.:  • dataset_split: The dataset split. `train`, `val`, `test`
            • config: The configuration (dict) object for system configuration
    Return: torchvision.transforms
    """
    if dataset_split not in ['train','val', 'test']:
        raise ValueError("Invalid dataset_split. Expected value to be `train`, `val`, `test` but got `{}`".format(dataset_split))
    
    if config is None:
        raise ValueError("`config` can not be None.")
    
    if dataset_split == 'train' or dataset_split == 'val':
        transform = T.Compose([T.Resize(config.image_size, interpolation=3),
                               T.Pad(10),
                               T.RandomCrop(config.image_size),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               T.Normalize(config.mean,config.std)])
    else: # this is for test
        transform = T.Compose([T.Resize(config.image_size, interpolation=3),
                               T.ToTensor(),
                               T.Normalize(config.mean,config.std)])
    return transform



# build the dataloader
def build_cuhkpedes_dataloader(dataset_split:str=None, config:dict=None):
    """Build the dataloader"""
    
    if dataset_split not in ['train','val', 'test']:
        raise ValueError("Invalid dataset_split. Expected value to be `train`, `val`, `test` but got `{}`".format(dataset_split))
    
    dataset = CUHKPEDES(config)

    if dataset_split == 'train':
        transform = get_transform('train',config)
        image_text_data = dataset.train
        shuffle = True
        num_classes = len(dataset.train_id_container)
    
    elif dataset_split == 'val':
        transform = get_transform('val',config)
        image_text_data = dataset.val
        shuffle = True
        num_classes = len(dataset.val_id_container)
    
    else: # for test
        transform = get_transform('test',config)
        image_text_data = dataset.test
        shuffle = False
        num_classes = len(dataset.test_id_container)
    
    data = ImageTextDataset(image_text_data,
                            transform,
                            tokenizer_type=config.tokenizer_type,
                            tokens_length_max=config.tokens_length_max)
        
    data_loader = DataLoader(data,
                             batch_size=config.batch_size,
                             shuffle=shuffle,
                             num_workers=config.num_workers,
                             collate_fn=collate)

    return data_loader, num_classes

