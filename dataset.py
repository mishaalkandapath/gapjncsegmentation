import os
import cv2
import numpy as np;
import torch 
import torchvision.transforms as transforms
import torchio as tio

class SliceDataset(torch.utils.data.Dataset):
    """ Dataset for 2D slices of 3D EM images
    
    Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augment (bool): whether to apply augmentations
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            augment=False
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.augment = augment
    
    def __getitem__(self, i):
        # read images and masks (3D grayscale images)
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)

        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        image = tio.ZNormalization()(image)

        # define preprocessing and augmentations
        mask_augment = tio.Compose([
            tio.RandomFlip(flip_probability=0.5)
            ])
        img_augment = tio.Compose([
            mask_augment,
            tio.RandomBlur(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomGamma(p=0.5)
        ])

    
        # apply augmentations, if any
        if self.augment:
            image = img_augment(image)
            mask = mask_augment(mask)
            
        # one-hot encode the mask (depth, height, width) --> (depth, height, width, num_classes=2)
        one_hot_mask = torch.nn.functional.one_hot(mask.squeeze(0).long(), num_classes=2)
        one_hot_mask = one_hot_mask.permute(3, 0, 1, 2).float() # (num_classes, depth, height, width)
        return image, one_hot_mask
        
    def __len__(self):
        return len(self.image_paths)