import os
import cv2
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
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
            image_dim = (512, 512),
            augmentation=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.image_dim = image_dim
    
    def __getitem__(self, i):
        # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 0 or 255
        
        # pad the image to make it square
        img_size = 512
        width, height = self.image_dim
        max_dim = max(img_size, width, height)
        pad_left = (max_dim-width)//2
        pad_right = max_dim-width-pad_left
        pad_top = (max_dim-height)//2
        pad_bottom = max_dim-height-pad_top
        
        # define transformations for preprocessing
        _transform_mask = [] # (for mask)
        _transform_mask.append(transforms.ToTensor())
        _transform_mask.append(transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), padding_mode='edge')) # pad with edge values
        _transform_mask.append(transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size=(img_size, img_size))) # nearest neighbor interpolation
        _transform_img = _transform_mask.copy() # (for image)
        _transform_img.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        
        # apply transformations
        mask = transforms.Compose(_transform_mask)(mask)
        image = transforms.Compose(_transform_img)(image)
        
        # apply augmentations, if any
        if self.augmentation:
            image = self.augmentation(image)
            mask = self.augmentation(mask)
            
        # one-hot encode the mask (ie. convert from 0, 255 to 0, 1)
        ont_hot_mask = F.one_hot(mask.long(), num_classes=2).squeeze().permute(2, 0, 1).float()
        return image, ont_hot_mask
        
    def __len__(self):
        return len(self.image_paths)

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, device=torch.device("cpu")):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.device = device
        self.alpha = alpha.to(device)
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        targets = targets.to(torch.int64)
        loss = self.alpha[targets.view(-1, 512*512)].view(-1, 512, 512) * pt ** self.gamma * bce_loss
        return loss.mean() 

def checkpoint(model, optimizer, epoch, loss, path):
    """ Save model checkpoint
    
    Args:
        model (nn.Module): model to save
        optimizer (torch.optim): optimizer to save
        epoch (int): epoch number
        loss (float): loss value
        path (str): path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """ Load model checkpoint
    
    Args:
        model (nn.Module): model to load checkpoint
        optimizer (torch.optim): optimizer to load checkpoint
        path (str): path to load the checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

def crop_image(image, target_image_dims):
    """ Crop the image to the target image dimensions
    
    Args:
        image (np.array): image to crop
        target_image_dims (Tuple[int, int]): target image dimensions 
    """

    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

def find_centroids(segmented_img):
    """ Find the centroids of the segmented image
    
    Args:
        segmented_img (np.array): segmented image
    """
    centroids = []
    cont, hierarchy = cv2.findContours(segmented_img, 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))
    
    return centroids

def get_subset(dataset, subset_size):
    """Get a subset of the dataset
    
    Args:
        dataset (CaImagesDataset): dataset to get subset from
        subset_size (int): size of the subset
        
    Returns:
        subset (CaImagesDataset): subset of the dataset
    """
    # get a random subset of the dataset
    subset = torch.utils.data.Subset(dataset, random.sample(range(len(dataset)), subset_size))
    return subset