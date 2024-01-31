"""
Utilities for managing datasets
"""
# System modules
import os
import sys
import random

# Add project base directory to system path
PROJ_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0, PROJ_BASE_DIR) 

# Application modules
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def calculate_mean_std(loader):
    """ Calculates mean and stds of a particular dataset """
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


class CustomResizeAndPadVanilla():
    def __init__(self, desired_size=512):
        super(CustomResizeAndPad, self).__init__()
        self.desired_size = desired_size

    def __call__(self, img):
        # Calculate the new size, maintaining aspect ratio
        old_size = img.size
        ratio = float(self.desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # Resize the image
        img = img.resize(new_size, Image.ANTIALIAS)

        # Create a new image with black background and paste the resized image onto it
        new_img = Image.new("RGB", (self.desired_size, self.desired_size))
        new_img.paste(img, ((self.desired_size - new_size[0]) // 2,
                            (self.desired_size - new_size[1]) // 2))

        return new_img
    

class CustomResizeAndPadDifferentColor():
    def __init__(self, desired_size=512):
        """
        Initialize the resize and pad class.

        :param desired_size: The target size for both dimensions of the image.
        """
        super(CustomResizeAndPad, self).__init__()
        self.desired_size = desired_size

    def __call__(self, img):
        """
        Resize and pad the given image with a randomly generated padding color.

        :param img: The image to resize and pad, should be a PIL Image object.
        :return: The resized and padded image.
        """
        # Validate input
        if not isinstance(img, Image.Image):
            raise TypeError("The provided input is not a PIL Image object.")

        # Calculate the new size, maintaining aspect ratio
        old_size = img.size
        ratio = float(self.desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # Resize the image
        img = img.resize(new_size, Image.ANTIALIAS)

        # Generate a random color for padding
        padding_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Create a new image with the randomly generated background color and paste the resized image onto it
        new_img = Image.new("RGB", (self.desired_size, self.desired_size), padding_color)
        new_img.paste(img, ((self.desired_size - new_size[0]) // 2,
                            (self.desired_size - new_size[1]) // 2))

        return new_img

class CustomResizeAndPad():
    def __init__(self, desired_size=512):
        """
        Initialize the resize and pad class.

        :param desired_size: The target size for both dimensions of the image.
        """
        super(CustomResizeAndPad, self).__init__()
        self.desired_size = desired_size

    def __call__(self, img):
        """
        Resize and randomly position the image on a padded background with a 
        randomly generated padding color.

        :param img: The image to resize and pad, should be a PIL Image object.
        :return: The resized and padded image.
        """
        # Validate input
        if not isinstance(img, Image.Image):
            raise TypeError("The provided input is not a PIL Image object.")

        # Calculate the new size, maintaining aspect ratio
        old_size = img.size
        ratio = float(self.desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # Resize the image
        img = img.resize(new_size, Image.ANTIALIAS)

        # Generate a random color for padding
        padding_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Create a new image with the randomly generated background color
        new_img = Image.new("RGB", (self.desired_size, self.desired_size), padding_color)

        # Calculate random top-left corner for pasting
        max_x = self.desired_size - new_size[0]
        max_y = self.desired_size - new_size[1]
        top_left_x = random.randint(0, max_x)
        top_left_y = random.randint(0, max_y)

        # Paste the resized image onto the new image at a random position
        new_img.paste(img, (top_left_x, top_left_y))

        return new_img