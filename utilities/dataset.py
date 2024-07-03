"""
dataset.py: Contains the SliceDataset class for loading 3D subvolumes of 3D EM images
"""
import os
import cv2
import numpy as np;
import torch 
import torchvision.transforms as transforms
import torchio as tio

class SliceDatasetMultipleFolders(torch.utils.data.Dataset):
    """ Dataset for 2D slices of 3D EM images
    
    Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augment (bool): whether to apply augmentations
    """
    def __init__(
            self, 
            images_dir_lst, 
            masks_dir_lst,
            augment=False,
            downsample_factor=1
    ):
        
        self.image_paths = []
        self.mask_paths = []
        self.downsample_factor=downsample_factor
        self.augment = augment
        for images_dir in images_dir_lst:
            self.image_paths.extend([os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))])
        for masks_dir in masks_dir_lst:   
            self.mask_paths.extend([os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))])
    
    def __getitem__(self, i):
        """ 
        Get image and mask at index i
        
        Note: for original image, 0 is gap junction, 1 is not gap junction
        
        Returns:
            image (torch.Tensor): image tensor, shape (1, depth, height, width)
            one_hot_mask (torch.Tensor): one-hot encoded mask tensor, shape (num_classes=2, depth, height, width)
            - one_hot_mask[0] is the background class, or not gap junction class (1 if not gap junction)
            - one_hot_mask[1] is the foreground class, or gap junction class (1 if gap junction)
            USE ONE_HOT_MASK[1]
        """
        # read images and masks (3D grayscale images)
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)

        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask[mask!=0]=1
            
        if self.downsample_factor > 1:
            image = torch.nn.MaxPool3d((1, self.downsample_factor, self.downsample_factor), (1, self.downsample_factor, self.downsample_factor))(image)
            mask = torch.nn.MaxPool3d((1, self.downsample_factor, self.downsample_factor), (1, self.downsample_factor, self.downsample_factor))(mask)
        if torch.std(image) == 0:
            print(f"Image at index {i} has zero standard deviation, skipping normalization")
        else:
            image = tio.ZNormalization()(image)
    
        # apply augmentations, if any
        if self.augment:
            # Apply the flip transformation to the subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask)
            )
            all_transforms = [
                tio.RandomFlip(axes=0, flip_probability=0.5),
                tio.RandomFlip(axes=1, flip_probability=0.5),
                tio.RandomFlip(axes=2, flip_probability=0.5),
                tio.RandomAffine(scales=(0,0,0), degrees=(360,0,0), translation=(0,100,100)) # translate and rotate
            ]
            
            for transform in all_transforms:
                subject = transform(subject)
            image = subject.image.tensor
            mask = subject.mask.tensor
            
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
            augment=False,
            downsample_factor=1
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.augment = augment
        self.downsample_factor=downsample_factor
    
    def __getitem__(self, i):
        """ 
        Get image and mask at index i
        
        Note: for original image, 0 is gap junction, 1 is not gap junction
        
        Returns:
            image (torch.Tensor): image tensor, shape (1, depth, height, width)
            one_hot_mask (torch.Tensor): one-hot encoded mask tensor, shape (num_classes=2, depth, height, width)
            - one_hot_mask[0] is the background class, or not gap junction class (1 if not gap junction)
            - one_hot_mask[1] is the foreground class, or gap junction class (1 if gap junction)
            USE ONE_HOT_MASK[1]
        """
        # read images and masks (3D grayscale images)
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)

        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask[mask!=0]=1
        
        if self.downsample_factor > 1:
            image = torch.nn.MaxPool3d(self.downsample_factor)(image)
            mask = torch.nn.MaxPool3d(self.downsample_factor)(mask)
            
        if torch.std(image) == 0:
            print(f"Image at index {i} has zero standard deviation, skipping normalization")
        else:
            image = tio.ZNormalization()(image)
    
        # apply augmentations, if any
        if self.augment:
            # Apply the flip transformation to the subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask)
            )
            flip_transform = tio.RandomFlip(axes=0, flip_probability=0.5)
            flipped_subject = flip_transform(subject)
            flip_transform = tio.RandomFlip(axes=1, flip_probability=0.5)
            flipped_subject = flip_transform(flipped_subject)
            flip_transform = tio.RandomFlip(axes=2, flip_probability=0.5)
            flipped_subject = flip_transform(flipped_subject)
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
    
class SliceDatasetWithMemb(torch.utils.data.Dataset):
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
            cellmasks_dir,
            augment=False,
            downsample_factor=1
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.cellmask_paths = [os.path.join(cellmasks_dir, image_id) for image_id in sorted(os.listdir(cellmasks_dir))]
        self.augment = augment
        self.downsample_factor=downsample_factor
    
    def __getitem__(self, i):
        """ 
        Get image and mask at index i
        
        Note: for original image, 0 is gap junction, 1 is not gap junction
        
        Returns:
            image (torch.Tensor): image tensor, shape (1, depth, height, width)
            one_hot_mask (torch.Tensor): one-hot encoded mask tensor, shape (num_classes=2, depth, height, width)
            - one_hot_mask[0] is the background class, or not gap junction class (1 if not gap junction)
            - one_hot_mask[1] is the foreground class, or gap junction class (1 if gap junction)
            USE ONE_HOT_MASK[1]
        """
        # read images and masks (3D grayscale images)
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)
        cellmask = np.load(self.cellmask_paths[i]) # each pixel is 0 or 1, shape (1, height, width)
        # combmask = np.concatenate((mask[np.newaxis, ...], cellmask[np.newaxis, ...]), axis=0) # (2, 3, 512, 512)
        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        cellmask = torch.tensor(cellmask).float().unsqueeze(0) 
        # combmask[combmask!=0]=1
        if self.downsample_factor > 1:
            image = torch.nn.MaxPool3d(self.downsample_factor)(image)
            mask = torch.nn.MaxPool3d(self.downsample_factor)(mask)

            
        if torch.std(image) == 0:
            print(f"Image at index {i} has zero standard deviation, skipping normalization")
        else:
            image = tio.ZNormalization()(image)
    
        # apply augmentations, if any
        if self.augment:
            # Apply the flip transformation to the subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask),
                cellmask=tio.LabelMap(tensor=cellmask)
            )
            flip_transform = tio.RandomFlip(axes=0, flip_probability=0.5)
            flipped_subject = flip_transform(subject)
            flip_transform = tio.RandomFlip(axes=1, flip_probability=0.5)
            flipped_subject = flip_transform(flipped_subject)
            flip_transform = tio.RandomFlip(axes=2, flip_probability=0.5)
            flipped_subject = flip_transform(flipped_subject)
            image = flipped_subject.image.tensor
            mask = flipped_subject.mask.tensor
            cellmask = flipped_subject.cellmask.tensor

            # Define additional transformations for the image
            additional_transforms = tio.Compose([
                tio.RandomBlur(p=0.5),
                tio.RandomNoise(p=0.5),
                tio.RandomGamma(p=0.5)
            ])

            # Apply the additional transformations to the flipped image
            image = additional_transforms(image)
        combmask = torch.cat((mask, cellmask), dim=0)
        combmask[combmask!=0]=1
            
        # one-hot encode the mask (depth, height, width) --> (depth, height, width, num_classes=2)
        # one_hot_mask = torch.nn.functional.one_hot(combmask.squeeze(0).long(), num_classes=2)
        # one_hot_mask = one_hot_mask.permute(3, 0, 1, 2).float() # (num_classes, depth, height, width)
        return image, combmask
        
    def __len__(self):
        return len(self.image_paths)
    
    
class SliceDatasetWithMembThreeClasses(torch.utils.data.Dataset):
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
            cellmasks_dir,
            augment=False
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.cellmask_paths = [os.path.join(cellmasks_dir, image_id) for image_id in sorted(os.listdir(cellmasks_dir))]
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
            USE ONE_HOT_MASK[1]
        """
        # read images and masks (3D grayscale images)
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)
        cellmask = np.load(self.cellmask_paths[i]) # each pixel is 0 or 1, shape (1, height, width)
        # combmask = np.concatenate((mask[np.newaxis, ...], cellmask[np.newaxis, ...]), axis=0) # (2, 3, 512, 512)
        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        cellmask = torch.tensor(cellmask).float().unsqueeze(0) 
        # combmask[combmask!=0]=1
        image = tio.ZNormalization()(image)
    
        # apply augmentations, if any
        if self.augment:
            # Apply the flip transformation to the subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                mask=tio.LabelMap(tensor=mask),
                cellmask=tio.LabelMap(tensor=cellmask)
            )
            flip_transform = tio.RandomFlip(axes=0, flip_probability=0.5)
            flipped_subject = flip_transform(subject)
            flip_transform = tio.RandomFlip(axes=1, flip_probability=0.5)
            flipped_subject = flip_transform(flipped_subject)
            flip_transform = tio.RandomFlip(axes=2, flip_probability=0.5)
            flipped_subject = flip_transform(flipped_subject)
            image = flipped_subject.image.tensor
            mask = flipped_subject.mask.tensor
            cellmask = flipped_subject.cellmask.tensor

            # Define additional transformations for the image
            additional_transforms = tio.Compose([
                tio.RandomBlur(p=0.5),
                tio.RandomNoise(p=0.5),
                tio.RandomGamma(p=0.5)
            ])

            # Apply the additional transformations to the flipped image
            image = additional_transforms(image)
            
        # one-hot encode the mask (depth, height, width) --> (depth, height, width, num_classes=2)
        mask = torch.nn.functional.one_hot(mask.squeeze(0).long(), num_classes=2)
        mask = mask.permute(3, 0, 1, 2).float() # (num_classes, depth, height, width)
        combmask = torch.cat((mask, cellmask), dim=0)
        combmask[combmask!=0]=1
        return image, combmask
        
    def __len__(self):
        return len(self.image_paths)
    
    

class SliceDatasetWithFilename(torch.utils.data.Dataset):
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
            augment=False,
            downsample_factor=1,
            downsample_mask=True
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir)) if image_id.endswith(".npy")]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir)) if image_id.endswith(".npy")]
        self.augment = augment
        self.downsample_factor = downsample_factor
        self.downsample_mask = downsample_mask
    
    def __getitem__(self, i):
        """ 
        Get image and mask at index i
        
        Note: for original image, 0 is gap junction, 1 is not gap junction
        
        Returns:
            image (torch.Tensor): image tensor, shape (1, depth, height, width)
            one_hot_mask (torch.Tensor): one-hot encoded mask tensor, shape (num_classes=2, depth, height, width)
            - one_hot_mask[0] is the background class, or not gap junction class (1 if not gap junction)
            - one_hot_mask[1] is the foreground class, or gap junction class (1 if gap junction)
            USE ONE_HOT_MASK[1]
        """
        # read images and masks (3D grayscale images)
        file_name = os.path.basename(self.image_paths[i])
        file_name = os.path.splitext(file_name)[0]
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)

        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask[mask!=0]=1
        
        # Check if the standard deviation is zero
        if self.downsample_factor > 1:
            image = torch.nn.MaxPool3d((1, self.downsample_factor, self.downsample_factor), (1, self.downsample_factor, self.downsample_factor))(image)
            if self.downsample_mask:
                mask = torch.nn.MaxPool3d((1, self.downsample_factor, self.downsample_factor), (1, self.downsample_factor, self.downsample_factor))(mask)
        if torch.std(image) == 0:
            print(f"Image at index {i} has zero standard deviation, skipping normalization")
        else:
            image = tio.ZNormalization()(image)
        return image, mask, file_name
        
    def __len__(self):
        return len(self.image_paths)
    

class SliceDatasetWithFilenameAllSubfolders(torch.utils.data.Dataset):
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
            augment=False,
            downsample_factor=1
    ):
        
        self.image_paths = []
        self.mask_paths = []

        for root, dirs, files in os.walk(images_dir):
            for file in files:
                self.image_paths.append(os.path.join(root, file))

        for root, dirs, files in os.walk(masks_dir):
            for file in files:
                self.mask_paths.append(os.path.join(root, file))

        self.image_paths.sort()
        self.mask_paths.sort()
        self.augment = augment
        self.downsample_factor = downsample_factor
    
    def __getitem__(self, i):
        """ 
        Get image and mask at index i
        
        Note: for original image, 0 is gap junction, 1 is not gap junction
        
        Returns:
            image (torch.Tensor): image tensor, shape (1, depth, height, width)
            one_hot_mask (torch.Tensor): one-hot encoded mask tensor, shape (num_classes=2, depth, height, width)
            - one_hot_mask[0] is the background class, or not gap junction class (1 if not gap junction)
            - one_hot_mask[1] is the foreground class, or gap junction class (1 if gap junction)
            USE ONE_HOT_MASK[1]
        """
        # read images and masks (3D grayscale images)
        file_name = os.path.basename(self.image_paths[i])
        file_name = os.path.splitext(file_name)[0]
        image = np.load(self.image_paths[i]) # each pixel is 0-255, shape (depth, height, width)
        mask = np.load(self.mask_paths[i]) # each pixel is 0 or 1, shape (depth, height, width)

        # convert to tensor
        image = torch.tensor(image).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask = torch.tensor(mask).float().unsqueeze(0) # add channel dimension (depth, height, width) --> (1, depth, height, width)
        mask[mask!=0]=1
        # Check if the standard deviation is zero
        if self.downsample_factor > 1:
            image = torch.nn.MaxPool3d((1, self.downsample_factor, self.downsample_factor), (1, self.downsample_factor, self.downsample_factor))(image)
            mask = torch.nn.MaxPool3d((1, self.downsample_factor, self.downsample_factor), (1, self.downsample_factor, self.downsample_factor))(mask)
            
        if torch.std(image) == 0:
            print(f"Image at index {i} has zero standard deviation, skipping normalization")
        else:
            image = tio.ZNormalization()(image)
        return image, mask, file_name
        
    def __len__(self):
        return len(self.image_paths)