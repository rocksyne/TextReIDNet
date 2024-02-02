"""
Doc.: Miscellaneous utilities for codebase
"""

# System modules
import os
import random
import logging

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



def setup_logger(name:str=None, log_file_path:str=None, write_mode:str='overwrite', level:logging=logging.INFO):
    """
    Doc.:   Function to setup as many loggers as desired.
            The log files are cleared every time the 

    Args.:  • config: configuration parameters
            • name: name of the logger
            • log_file_path: file path to the log file
            • write_mode: eaither append to existing file or overwrite. Values: `overwrite` or `append`
            • level: logging level

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

    return logger