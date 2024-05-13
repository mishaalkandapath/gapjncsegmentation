import os
import cv2
import random
import re
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Tuple


def get_z_y_x(file_name, pattern) -> Tuple[int, int, int]:
    """ Get z, y, x from file name using pattern

    Args:
        file_name (str): file name (can be full path)
        pattern (str): pattern to extract z, y, x from file name
        
    Returns:
        z: int, z coordinate
        y: int, y coordinate
        x: int, x coordinate
    """
    file_name = os.path.basename(file_name)
    match = re.match(pattern, file_name)
    if match:
        if len(match.groups()) != 3:
            return None
        z, y, x = match.groups()
        return int(z), int(y), int(x)
    else:
        return None
    
def get_img_by_coords(z, y, x, img_files, img_pattern) -> np.array:
    """ Get image from img_files by z, y, x
    
    Args:
        z: int, z coordinate
        y: int, y coordinate
        x: int, x coordinate
        img_files: list of str, paths to images
        img_pattern: str, pattern to extract z, y, x from image file name
        
    Returns:
        img: np.array, image whose file path contains z, y, x (ie. slice at z, y, x)
    """
    for i in range(len(img_files)):
        img_file = img_files[i]
        z_, y_, x_ = get_z_y_x(img_file, img_pattern)
        if z == z_ and y == y_ and x == x_:
            return cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    return None

def get_3d_slice(z,y,x, img_files, mask_files, img_pattern, mask_pattern, depth=1, width=512, height=512) -> Tuple[np.array, np.array]:
    """ get 3d slice by z, y, x 
    
    -think of the original image as 3d volume, and each slice is a pixel
    -so we want a column of pixels, with the center pixel being the pixel at z, y, x, and sliding up and down along the z axis
    
    Args:
        z: int, z coordinate
        y: int, y coordinate
        x: int, x coordinate
        img_files: list of str, paths to images
        mask_files: list of str, paths to masks
        img_pattern: str, pattern to extract z, y, x from image file name
        mask_pattern: str, pattern to extract z, y, x from mask file name
        depth: int, how many slices to include above and below the center slice
        width: int, width of each slice
        height: int, height of each slice
        
    Returns:
        img_3d: np.array, 3d image volume (2*depth+1, width, height)
        mask_3d: np.array, 3d mask volume (2*depth+1, width, height)
    """
    img_3d = np.zeros((2*depth+1, width, height))
    mask_3d = np.zeros((2*depth+1, width, height))
    for i in range(-depth, depth+1):
        z_coord = z+i
        img = get_img_by_coords(z_coord, y, x, img_files, img_pattern)
        mask = get_img_by_coords(z_coord, y, x, mask_files, mask_pattern)
        if img is not None and mask is not None:
            img_3d[i+depth] = img
            mask_3d[i+depth] = mask
    return img_3d, mask_3d

def checkpoint(model, optimizer, epoch, loss, batch_size, lr, focal_loss_weights, path):
    """ Save model checkpoint
    
    Args:
        model (nn.Module): model to save
        optimizer (torch.optim): optimizer to save
        epoch (int): epoch number
        loss (float): loss value
        batch_size (int): batch size
        lr (float): learning rate
        focal_loss_weights (Tuple[float, float]): weights for focal loss
        path (str): path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'batch_size': batch_size,
        'lr': lr,
        'focal_loss_weights': focal_loss_weights
    }, path)

def load_checkpoint(model, optimizer, path):
    """ Load model checkpoint
    
    Args:
        model (nn.Module): model to load checkpoint
        optimizer (torch.optim): optimizer to load checkpoint
        path (str): path to load the checkpoint
        
    Returns:
        model (nn.Module): loaded model
        optimizer (torch.optim): loaded optimizer
        epoch (int): epoch number
        loss (float): loss value
        batch_size (int): batch size
        lr (float): learning rate
        focal loss weights (Tuple[float, float]): weights for focal loss
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    batch_size = checkpoint['batch_size']
    lr = checkpoint['lr']
    focal_loss_weights = checkpoint['focal_loss_weights']
    
    
    return model, optimizer, epoch, loss, batch_size, lr, focal_loss_weights

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