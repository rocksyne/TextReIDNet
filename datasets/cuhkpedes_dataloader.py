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
def build_cuhkpedes_dataloader(config:dict=None):
    """Build the dataloader"""
    
    dataset_object = CUHKPEDES(config)

    # lets do the train set
    train_transform = get_transform('train',config)
    train_data = ImageTextDataset(dataset_object.train,
                                  train_transform,
                                  tokenizer_type=config.tokenizer_type,
                                  tokens_length_max=config.tokens_length_max)
    train_data_loader = DataLoader(train_data,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   collate_fn=collate)
    train_num_classes = len(dataset_object.train_id_container)
    
    # lets do the validation set
    val_transform = get_transform('val',config) # same as train by the way
    val_data = ImageTextDataset(dataset_object.val,
                                  val_transform,
                                  tokenizer_type=config.tokenizer_type,
                                  tokens_length_max=config.tokens_length_max)
    val_data_loader = DataLoader(val_data,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   collate_fn=collate)
    val_num_classes = len(dataset_object.val_id_container)
    
    # lets do for test set
    test_transform = get_transform('test',config) # same as train by the way
    test_data = ImageTextDataset(dataset_object.test,
                                  test_transform,
                                  tokenizer_type=config.tokenizer_type,
                                  tokens_length_max=config.tokens_length_max)
    test_data_loader = DataLoader(test_data,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=config.num_workers,
                                   collate_fn=collate)
    test_num_classes = len(dataset_object.test_id_container)
    
    data = dict(train_data_loader=train_data_loader,
                train_num_classes=train_num_classes,
                val_data_loader=val_data_loader,
                val_num_classes=val_num_classes,
                test_data_loader=test_data_loader,
                test_num_classes=test_num_classes
                )
    
    return data

