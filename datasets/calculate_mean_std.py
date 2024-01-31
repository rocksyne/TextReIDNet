"""
Calculate the mean and stds of a particular dataset
"""
import os
import sys

# Add project base directory to system path
PROJ_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0, PROJ_BASE_DIR) 

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# application modules
from datasets.utils import calculate_mean_std
from datasets.utils import CustomResizeAndPad



if __name__ == '__main__':

    dataset_path = "/home/users/roagyeman/research/datasets/CUHK-PEDES/imgs"
    
    # Transformations - Convert image to PyTorch tensor
    transform = transforms.Compose([CustomResizeAndPad(512),
                                    transforms.ToTensor()])

    # Load the dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=18)
    
    # Calculate mean and std
    print("Calculating normalization values. Please wait...")
    mean, std = calculate_mean_std(dataloader)

    print(f'Mean: {mean}')
    print(f'Std: {std}')