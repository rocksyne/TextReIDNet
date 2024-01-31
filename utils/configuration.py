"""
January 2023
Doc: Application configuration file
"""

# system modules import
import os
import sys
import pathlib
import platform

# 3rd party library
from prettytable import PrettyTable

# some global configurations
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # project parent directory
sys.path.insert(0,PROJ_DIR) 
table = PrettyTable(['Configuration Parameter', 'Value'],hrules=1)
table.align['Configuration Parameter'] = 'l'  # Left-align column
table.align['Value'] = 'l'  # Left-align the column


class DotDict(dict):
    """
    Doc.: dot.notation access to dictionary attributes
    Ref.: https://stackoverflow.com/a/23689767/3901871
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def sys_configuration(server_type:str = platform.node(), show_configs:bool=False)->dict():
    """
    Doc.: configuration parameters for application
    Args.:  show_configs: show sytem configurations of not
    """
    if server_type not in ['PC-SIM','deeplearning','ultron','nano']:
        raise ValueError('server_type be must string and of value `PC-SIM`, `deeplearning`, or `ultron` \
                         Provided name is {}'.format(server_type))
    
    configs = dict()
    
    # Experimented with different tokenizers to get the maximum index of the tokens
    # Bert tokenizer: 29609
    # Simple tokenizer: 49407
    # Tiktoken (cl100k_base): 100155
    # Tiktoken (p50k_base): 50276
    # Tiktoken (r50k_base): 50240
    max_tokenizer_size:dict[str,int] = dict(bert = 29609,
                                            simple_tokenizer = 49407,
                                            tiktoken_cl100k = 100155,
                                            tiktoken_p50k = 50276,
                                            tiktoken_r50k = 50240)
    
    configs['server_type'] = server_type

    
    # training or server configuration parameters
    if configs['server_type'] == 'PC-SIM': 
        configs['CUHK_PEDES_dataset_parent_dir'] = "/home/users/roagyeman/research/datasets/CUHK-PEDES" # parent dir of the dataset
        configs['num_workers'] =  16# Use 16 CPU cores max
        configs['batch_size'] = 32
        configs['progress_bar_width']:int = 3
    
    elif configs['server_type'] == 'deeplearning': 
        configs['CUHK_PEDES_dataset_parent_dir'] = "/media/rockson/Data_drive/datasets/CUHK-PEDES"
        configs['num_workers'] = 6 # Use 6 CPU cores max
        configs['batch_size'] = 1
        configs['progress_bar_width']:int = 3

    elif configs['server_type'] == 'ultron':
        configs['CUHK_PEDES_dataset_parent_dir'] = "/datasets/CUHK-PEDES"
        configs['num_workers'] = 6 # Use 6 CPU cores max
        configs['batch_size'] = 16
        configs['progress_bar_width']:int = 3
    
    elif configs['server_type'] == 'nano':
        configs['CUHK_PEDES_dataset_parent_dir'] = "/datasets/CUHK-PEDES"
        configs['num_workers'] = 2 # Use 6 CPU cores max
        configs['batch_size'] = 1
        configs['progress_bar_width']:int = 1
        

    else:
        raise ValueError("`{}` is and unknown server name!")


    # dataset
    # configs["mean"] = [0.48145466, 0.4578275, 0.40821073] original (384, 128) from IRRA
    # configs["std"]  = [0.26862954, 0.26130258, 0.27577711] (384, 128) from IRAA
    configs["mean"] = [0.4604, 0.4503, 0.4436] # 512 normalization 0.4604, 0.4503, 0.4436
    configs["std"]  = [0.1980, 0.1974, 0.1976] # 512 normalization
    configs["caption_length_max"] = 100
    configs["img_size"]:tuple = (512, 512)
    configs["val_dataset"]:str = "test" # choose which data set to use for vaildation / evaluation. Values: `test` or `val`

    # Other system parameters 
    configs['device']:str  = "cuda" # both systems have 1 GPU only so forget about the IDs
    configs["part"]:int = 6
    configs['feature_length']:int = 1024
    configs['epoch']:int = 45 # number of epochs for which model should be trained
    configs['adam_alpha']:float = 0.90 # momentum term of adam
    configs['adam_beta']:float = 0.999 # momentum term of adam
    configs["epoch_decay"] = [20,40]
    configs["class_num"]:int = 11003
    configs['margin']:float = 0.2
    configs['last_lstm']:bool = False

    # Tokenizer types: bert, simple_tokenizer, tiktoken_cl100k, tiktoken_p50k, tiktoken_r50k
    configs['tokenizer_type']:str = 'bert'

    # get the appropriate vocabulary size relative the tokenizer type
    # Add +1 to account for array indexing related operations
    configs['vocab_size'] = max_tokenizer_size[configs['tokenizer_type']]+1 

    configs['save_path']:str = os.path.join(PROJ_DIR,'data','checkpoints')
    configs['lr'] = 0.001
    configs['patience'] = 3


    # show debug information
    if show_configs:
        for key, value in configs.items():
            table.add_row([key, value])
        print(table)

    return DotDict(configs)
