import os
import cv2
import numpy as np;
import torch 
import torchvision.transforms as transforms

class SliceDataset(torch.utils.data.Dataset):
    """ Dataset for 2D slices of 3D EM images
    
    Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (torchvision.transforms.Compose): data augmentations
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            image_dim = (5, 512, 512),
            augmentation=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.image_dim = image_dim
        self.augmentation = augmentation
    
    def __getitem__(self, i):
        # read images and masks (3D grayscale images)
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)
        
        # normalize image to have mean 0 and std 1
        image = (image - image.mean()) / image.std() 
        
        # convert to tensor
        image = torch.tensor(image).float()
        mask = torch.tensor(mask).float()
        
        # apply augmentations, if any
        if self.augmentation:
            image = self.augmentation(image)
            mask = self.augmentation(mask)
            
        # one-hot encode the mask (depth, height, width) --> (depth, height, width, num_classes=2)
        one_hot_mask = torch.nn.functional.one_hot(mask.long(), num_classes=2) # 0: depth, 1: height, 2: width, 3: num_classes
        one_hot_mask = one_hot_mask.permute(3, 0, 1, 2).float() # (num_classes, depth, height, width)
        return image, one_hot_mask
        
    def __len__(self):
        return len(self.image_paths)