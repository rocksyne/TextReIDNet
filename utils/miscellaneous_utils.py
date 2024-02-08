"""
Doc.: Miscellaneous utilities for codebase
"""

# System modules
import os
import random
import logging
import datetime

# 3rd party modules
import torch
import numpy as np


def set_seed(seed_value:int=3407):
    """
    Doc.: Set seed for for reproducibility
    Args.: seed_value: seed value 
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def setup_logger(name:str=None, 
                 log_file_path:str=None, 
                 write_mode:str='overwrite', 
                 level:logging=logging.INFO,
                 timestamp:str=None):
    """
    Doc.:   Function to setup as many loggers as desired.
            The log files are cleared every time the 

    Args.:  • config: configuration parameters
            • name: name of the logger
            • log_file_path: file path to the log file
            • write_mode: eaither append to existing file or overwrite. Values: `overwrite` or `append`
            • level: logging level
            • timestamp: log the timestamp also

    Returns: logger object
    """
    if name is None:
        raise ValueError("`name` can not be None")
    
    if log_file_path is None:
        raise ValueError("`log_file_path` can not be None")
    
    if level is None:
        raise ValueError("`level` can not be None")

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers according to the write mode
    if write_mode == 'overwrite':
        file_handler = logging.FileHandler(log_file_path, mode='w')
    elif write_mode == 'append':
        file_handler = logging.FileHandler(log_file_path, mode='a')
    else:
        raise NotImplementedError("No implementaion for `write_mode`={}".format(write_mode))
    
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    if timestamp:
        logger.info("")
        logger.info("Logging started at ",timestamp)
        logger.info("")

    return logger



def pad_tokens(tokens, tokens_length_max):
    """
    This entire code snippet is adapted from the URL below.
    https://github.com/xx-adeline/MFPE/blob/main/src/data/dataset.py#L65
    Enhancement: maske tensors contigeous for memory efficiency

    TODO: [x]Change variable names to make more applicable

    Args.:  tokens: Textual descriptions converted into tokens. Eg. [1165, 13, 564, 74, ..., 1167]
            tokens_length_max: The maximum number of tokens needed, eg, 100

    Return: list: padded tokens and the original length of the tokens before padding to tokens_length_max
    """
    tokens_length = len(tokens)
    tokens = torch.from_numpy(np.array(tokens)).view(-1).contiguous().float() # make contiguous
    if tokens_length < tokens_length_max:
        zero_padding = torch.zeros(tokens_length_max - tokens_length)
        tokens = torch.cat([tokens, zero_padding], 0)
    else:
        tokens = tokens[:tokens_length_max]
        tokens_length = tokens_length_max
    return tokens, tokens_length



def collate(batch):
    """Data collator for the dataloader"""
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