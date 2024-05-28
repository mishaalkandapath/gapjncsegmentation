""" 
Utility functions for visualization, processing data, and saving/loading models
"""
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
from matplotlib import pyplot as plt

# --- saving and loading models ---
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
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    batch_size = checkpoint['batch_size']
    lr = checkpoint['lr']
    focal_loss_weights = checkpoint['focal_loss_weights']
    
    
    return model, optimizer, epoch, loss, batch_size, lr, focal_loss_weights

# --- data processing ---
def create_train_valid_test_dir(data_dir):
    """ 
    Create folders for train, valid, and test data for ground truth and original images in data_dir
    """
    img_dir = os.path.join(data_dir, "original")
    mask_dir = os.path.join(data_dir, "ground_truth")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(os.path.join(img_dir, "train")):
        os.makedirs(os.path.join(img_dir, "train"))
    if not os.path.exists(os.path.join(img_dir, "valid")):
        os.makedirs(os.path.join(img_dir, "valid"))
    if not os.path.exists(os.path.join(img_dir, "test")):
        os.makedirs(os.path.join(img_dir, "test"))   
    if not os.path.exists(os.path.join(mask_dir, "train")):
        os.makedirs(os.path.join(mask_dir, "train"))
    if not os.path.exists(os.path.join(mask_dir, "valid")):
        os.makedirs(os.path.join(mask_dir, "valid"))
    if not os.path.exists(os.path.join(mask_dir, "test")):
        os.makedirs(os.path.join(mask_dir, "test"))
        
def visualize_3d_slice(img: np.array, fig_ax: plt.Axes, title: str = ""):
    """ 
    Takes in a 3d image of shape (depth, height, width) and plots each z-slice as a row of 2D images on the given axis.
    
    Args:
        img (np.array): 3D image to visualize (depth, height, width)
        fig_ax (plt.Axes): matplotlib axis to plot on
        title (str): title of the plot
        
    Sample Usage:
        fig, ax = plt.subplots(3, depth, figsize=(15, 5), num=1)
        visualize_3d_slice(input_img, ax[0], "Input")
        visualize_3d_slice(label_img, ax[1], "Ground Truth")
        visualize_3d_slice(pred_img, ax[2], "Prediction")
    """
    depth, width, height = img.shape
    for i in range(depth):
        fig_ax[i].imshow(img[i], cmap="gray")
    fig_ax[0].set_ylabel(title)

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
        result = get_z_y_x(img_file, img_pattern)
        if len(result) != 3:
            print("result", result)
            return None
        else:
            z_, y_, x_ = result
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

# --- evaluation metrics ---
def dice_coefficient(pred, target, smooth=1e-6):
    """ Calculate dice coefficient for binary segmentation
    
    Args:
        pred (torch.Tensor): predicted mask, shape (batch_size, 1, depth, height, width)
        target (torch.Tensor): target mask, shape (batch_size, 1, depth, height, width)
        smooth (float): smoothing factor to prevent division by zero
        
    Returns:
        dice (torch.Tensor): dice coefficient
    """
    pred = pred.view(-1) # flatten
    target = target.view(-1) # flatten
    intersection = (pred * target).sum() # (pred * target) is 1 if both are 1, 0 otherwise
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def get_confusion_matrix(pred, target):
    """ Calculate confusion matrix for binary segmentation
    
    [[true_positives, false_positives], [false_negatives, true_negatives]]
    
    Args:
        pred (torch.Tensor): predicted mask, shape (depth, height, width)
        target (torch.Tensor): target mask, shape (depth, height, width)
        
    Returns:
        confusion_matrix (torch.Tensor): confusion matrix
    """
    pred = pred.view(-1) # flatten
    target = target.view(-1) # flatten
    true_positives = torch.sum(pred * target) # pred and target are both 1  
    false_positives = torch.sum(pred * (1 - target)) # pred is 1, target is 0
    false_negatives = torch.sum((1 - pred) * target) # pred is 0, target is 1
    true_negatives = torch.sum((1 - pred) * (1 - target)) # pred and target are both 0
    return true_positives, false_positives, false_negatives, true_negatives

def get_iou(pred, target, smooth=1e-6):
    """ Calculate intersection over union for binary segmentation
    
    Args:
        pred (torch.Tensor): predicted mask, shape (batch_size, 1, depth, height, width)
        target (torch.Tensor): target mask, shape (batch_size, 1, depth, height, width)
        smooth (float): smoothing factor to prevent division by zero
        
    Returns:
        iou (torch.Tensor): intersection over union
    """
    pred = pred.view(-1) # flatten
    target = target.view(-1) # flatten
    intersection = (pred * target).sum() # (pred * target) is 1 if both are 1, 0 otherwise
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def get_precision(pred, target, smooth=1e-6):
    """ Calculate precision for binary segmentation
    
    Args:
        pred (torch.Tensor): predicted mask, shape (batch_size, 1, depth, height, width)
        target (torch.Tensor): target mask, shape (batch_size, 1, depth, height, width)
        smooth (float): smoothing factor to prevent division by zero
        
    Returns:
        precision (torch.Tensor): precision
    """
    pred = pred.view(-1)
    target = target.view(-1)
    true_positives = (pred * target).sum()
    false_positives = pred.sum() - true_positives
    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
    return precision

def get_recall(pred, target, smooth=1e-6):
    """ Calculate recall for binary segmentation
    
    Args:
        pred (torch.Tensor): predicted mask, shape (batch_size, 1, depth, height, width)
        target (torch.Tensor): target mask, shape (batch_size, 1, depth, height, width)
        smooth (float): smoothing factor to prevent division by zero
        
    Returns:
        recall (torch.Tensor): recall
    """
    pred = pred.view(-1)
    target = target.view(-1)
    true_positives = (pred * target).sum()
    false_negatives = target.sum() - true_positives
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)
    return recall

def get_f1_score(pred, target, smooth=1e-6):
    """ Calculate f1 score for binary segmentation
    
    Args:
        pred (torch.Tensor): predicted mask, shape (batch_size, 1, depth, height, width)
        target (torch.Tensor): target mask, shape (batch_size, 1, depth, height, width)
        smooth (float): smoothing factor to prevent division by zero
        
    Returns:
        f1 (torch.Tensor): f1 score
    """
    prec = precision(pred, target, smooth)
    rec = recall(pred, target, smooth)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

def get_accuracy(pred, target):
    """ Calculate accuracy for binary segmentation
    
    Args:
        pred (torch.Tensor): predicted mask, shape (batch_size, 1, depth, height, width)
        target (torch.Tensor): target mask, shape (batch_size, 1, depth, height, width)
        
    Returns:
        accuracy (torch.Tensor): accuracy
    """
    pred = pred.view(-1)
    target = target.view(-1)
    correct = (pred == target).sum()
    total = pred.shape[0]
    accuracy = correct / total
    return accuracy