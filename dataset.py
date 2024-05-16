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
        """ 
        Get image and mask at index i
        
        Note: for original image, 0 is gap junction, 1 is not gap junction
        
        Returns:
            image (torch.Tensor): image tensor, shape (1, depth, height, width)
            one_hot_mask (torch.Tensor): one-hot encoded mask tensor, shape (num_classes=2, depth, height, width)
            - one_hot_mask[0] is the background class, or not gap junction class (1 if not gap junction)
            - one_hot_mask[1] is the foreground class, or gap junction class (1 if gap junction)
        """
        # read images and masks (3D grayscale images)
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)

        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        image = tio.ZNormalization()(image)
    
        # apply augmentations, if any
        if self.augment:
            # Apply the flip transformation to the subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask)
            )
            flip_transform = tio.RandomFlip(flip_probability=1)
            flipped_subject = flip_transform(subject)
            image = flipped_subject.image.tensor
            mask = flipped_subject.mask.tensor

            # Define additional transformations for the image
            additional_transforms = tio.Compose([
                tio.RandomBlur(p=0.5),
                tio.RandomNoise(p=0.5),
                tio.RandomGamma(p=0.5)
            ])

            # Apply the additional transformations to the flipped image
            image = additional_transforms(image)
            
        # one-hot encode the mask (depth, height, width) --> (depth, height, width, num_classes=2)
        one_hot_mask = torch.nn.functional.one_hot(mask.squeeze(0).long(), num_classes=2)
        one_hot_mask = one_hot_mask.permute(3, 0, 1, 2).float() # (num_classes, depth, height, width)
        return image, one_hot_mask
        
    def __len__(self):
        return len(self.image_paths)