import sys
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# some global configurations
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # project parent directory
sys.path.insert(0,PROJ_DIR) 

from utils.configuration import sys_configuration
from datasets.utils import CustomResizeAndPad

from datasets.bases import ImageDataset, TextDataset, ImageTextDataset

from datasets.cuhkpedes import CUHKPEDES

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # project parent directory
sys.path.insert(0,PROJ_DIR)
config = sys_configuration()


def build_transforms(img_size=(384, 128), aug=False):
    height, width = img_size

    print("We built the new transform")

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((384, 128), interpolation=3),
            T.Pad(10),
            T.RandomCrop((384, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.Resize((384, 128), interpolation=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        elif isinstance(v[0], str):
             batch_tensor_dict.update({k: v})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def build_dataloader(dataset_name=None, tranforms=None, mode='training'):

    supported_datasets = ["CUHKPEDES", "ICFGPEDES", "RSTPReid"]
    if dataset_name not in supported_datasets:
        raise ValueError("`{}` is not a supported dataset type".format(dataset))

    # check the dataset
    if dataset_name == "CUHKPEDES":
        dataset = CUHKPEDES()
    
    else:
        raise NotImplementedError("Implementation pending...")

    num_workers = config.num_workers
    num_classes = len(dataset.train_id_container)
    
    if mode == "training":
        train_transforms = build_transforms(img_size=config.img_size,aug=True)
        val_transforms = build_transforms(img_size=config.img_size,aug=False)


        train_set = ImageTextDataset(dataset.train,
                                    train_transforms,
                                    tokenizer_type=config.tokenizer_type,
                                    caption_length_max=config.caption_length_max)
        

        train_loader = DataLoader(train_set,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    collate_fn=collate)

        # use test set as validate set
        ds = dataset.val if config.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'], ds['captions'], tokenizer_type=config.tokenizer_type, caption_length_max=config.caption_length_max)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes
    

    elif mode == "testing":
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=config.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'], ds['captions'], tokenizer_type=config.tokenizer_type, caption_length_max=config.caption_length_max)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        
        return test_img_loader, test_txt_loader, num_classes
    
    else:
        raise ValueError("Invalid mode. Please use `training` or `testing`")
