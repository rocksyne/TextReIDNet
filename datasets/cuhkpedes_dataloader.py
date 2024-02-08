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
from datasets.bases import ImageTextDataset, ImageDataset, TextDataset


def get_transform(dataset_split:str=None, config:dict=None):
    """
    Doc.:   Get the appropriate transform
    Args.:  • dataset_split: The dataset split. `train`, `val`, `test`
            • config: The configuration (dict) object for system configuration
    Return: torchvision.transforms
    """
    if dataset_split not in ['train','inference']:
        raise ValueError("Invalid dataset_split. Expected value to be `train`, `inference` but got `{}`".format(dataset_split))
    
    if config is None:
        raise ValueError("`config` can not be None.")
    
    if dataset_split == 'train':
        transform = T.Compose([T.Resize(config.image_size, T.InterpolationMode.BICUBIC),
                               T.Pad(10),
                               T.RandomCrop(config.image_size),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               T.Normalize(config.mean,config.std)])
        
    else: # this is for val and test
        transform = T.Compose([T.Resize(config.image_size, T.InterpolationMode.BICUBIC),
                               T.ToTensor(),
                               T.Normalize(config.mean,config.std)])
        
    return transform



# build the dataloader
def build_cuhkpedes_dataloader(config:dict=None):
    """Build the dataloader"""
    
    dataset_object = CUHKPEDES(config)

    # lets do the train set
    train_transform = get_transform('train',config)
    inference_transform = get_transform('inference',config)
    train_num_classes = len(dataset_object.train_id_container)

    train_set = ImageTextDataset(dataset_object.train,
                                 train_transform,
                                 tokenizer_type=config.tokenizer_type,
                                 tokens_length_max=config.tokens_length_max)
    
    train_data_loader = DataLoader(train_set,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   collate_fn=collate)

    ds = dataset_object.val if config.model_testing_data_split == 'val' else dataset_object.test
    inference_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],inference_transform)
    inference_txt_set = TextDataset(ds['caption_pids'], ds['captions'], tokenizer_type=config.tokenizer_type, tokens_length_max=config.tokens_length_max)

    inference_img_loader = DataLoader(inference_img_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers)
    inference_txt_loader = DataLoader(inference_txt_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers)
    
    return train_data_loader, inference_img_loader, inference_txt_loader, train_num_classes

