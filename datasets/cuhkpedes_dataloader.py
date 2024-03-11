"""
Doc.:   Codebase adapted from https://github.com/anosorae/IRRA/tree/main
"""

# 3rd party modules
from torch.utils.data import DataLoader

# Application modules
from utils.miscellaneous_utils import collate, get_transform
from datasets.cuhkpedes import CUHKPEDES
from datasets.bases import ImageTextDataset, ImageDataset, TextDataset


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

