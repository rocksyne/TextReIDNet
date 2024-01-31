import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class ResizeWithPadding(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)
        self.interpolation = interpolation
        self.size = size

    def __call__(self, img):
        old_width, old_height = img.size
        new_width, new_height = self.size

        # Determine which axis is the longest
        if old_width >= old_height:
            # Resize based on the width and maintain aspect ratio
            scale = new_width / old_width
            new_size = (new_width, int(old_height * scale))
        else:
            # Resize based on the height and maintain aspect ratio
            scale = new_height / old_height
            new_size = (int(old_width * scale), new_height)

        # Resize the image using the specified interpolation
        img = transforms.functional.resize(img, new_size, self.interpolation)

        # Create a new blank image with the desired size and paste the resized image onto it
        padded_img = Image.new('RGB', self.size, (0, 0, 0))
        left = (new_width - img.width) // 2
        top = (new_height - img.height) // 2
        padded_img.paste(img, (left, top))

        return padded_img

