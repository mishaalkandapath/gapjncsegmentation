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

def visualize_3d_slice(img: np.array, fig_ax: plt.Axes, title: str = "", cmap="gray"):
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
    depth = img.shape[0]
    for i in range(depth):
        fig_ax[i].imshow(img[i], cmap=cmap)
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

def get_z(file_name, pattern):
    """ get z from file name (uses basename of file_name)"""
    file_name = os.path.basename(file_name)
    match = re.match(pattern, file_name)
    if match:
        z = match.groups()[0]
        return int(z)
    else:
        return None

def get_img_by_z(z, img_files, img_pattern):
    """ get path of image by z, y, x """
    for i in range(len(img_files)):
        img_file = img_files[i]
        try:
            z_ = get_z(img_file, img_pattern)
        except:
            continue
        if z == z_:
            return cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    return None

def get_3d_slice_by_z(z, img_files, img_pattern, depth_padding=1, width=512, height=512):
    img_3d = np.zeros((2*depth_padding+1, width, height))
    for i in range(-depth_padding, depth_padding+1):
        z_coord = z+i
        img = get_img_by_z(z_coord, img_files, img_pattern)
        if img is not None:
            img_3d[i+depth_padding] = img
    return img_3d


# --- expand/shrink 3D blobs ---
def shrink_binary_mask_3d(mask, kernel_size=(3,3,3)):
    """
    Shrinks the 3D binary mask by eroding it by a padding of 1.

    Parameters:
    - mask: torch.Tensor of shape (D, H, W) containing binary values (0 and 1)

    Returns:
    - shrunk_mask: torch.Tensor of the same shape as mask with shrunk blobs
    """
    # Create a 3D structuring element (kernel)
    depth_kernel, height_kernel, width_kernel = kernel_size
    if depth_kernel % 2 == 0: depth_kernel += 1 # Ensure the kernel size is odd (if not, add 1)
    if height_kernel % 2 == 0: height_kernel += 1
    if width_kernel % 2 == 0: width_kernel += 1
    structuring_element = torch.ones((1, 1, depth_kernel, height_kernel, width_kernel), dtype=torch.float32)

    # Apply dilation on the inverted mask (equivalent to erosion on the original mask)
    inverted_mask = 1 - mask.unsqueeze(0).unsqueeze(0).float() # Invert the mask for erosion (1s become 0s and 0s become 1s)
    eroded_mask = F.conv3d(inverted_mask, structuring_element, padding=(depth_kernel//2, height_kernel//2, width_kernel//2))
    eroded_mask = (eroded_mask < structuring_element.sum()).float() # Threshold the result to obtain a binary mask again
    shrunk_mask = 1 - eroded_mask # Invert back to get the eroded original mask
    shrunk_mask = shrunk_mask.squeeze(0).squeeze(0) # Remove batch and channel dimensions

    return shrunk_mask

def expand_binary_mask_3d(mask, kernel_size=(3,3,3)):
    """
    Expands the 3D binary mask using morphological dilation.

    Parameters:
    - mask: torch.Tensor of shape (D, H, W) containing binary values (0 and 1)
    - kernel_size: (int,int, int) size of the structuring element (must be odd)

    Returns:
    - expanded_mask: torch.Tensor of the same shape as mask with expanded blobs
    """
    # Create a 3D structuring element (kernel)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_kernel, height_kernel, width_kernel = kernel_size
    if depth_kernel % 2 == 0: depth_kernel += 1 # Ensure the kernel size is odd (if not, add 1)
    if height_kernel % 2 == 0: height_kernel += 1
    if width_kernel % 2 == 0: width_kernel += 1
    structuring_element = torch.ones((1, 1, depth_kernel, height_kernel, width_kernel), dtype=torch.float32)
    structuring_element = structuring_element.to(DEVICE)

    # Apply dilation using 3D convolution
    mask = mask.unsqueeze(0).unsqueeze(0).float() # Add batch and channel dimensions to the mask
    expanded_mask = F.conv3d(mask, structuring_element, padding=(depth_kernel//2, height_kernel//2, width_kernel//2))
    expanded_mask = (expanded_mask > 0).float() # Threshold the result to obtain a binary mask again
    expanded_mask = expanded_mask.squeeze(0).squeeze(0) # Remove batch and channel dimensions

    return expanded_mask

def get_colored_image(image, color_map=None):
    # Define the default color map
    if color_map is None:
        color_map = {
            0: [0, 0, 0],  # Black (TN)
            1: [1, 0, 0],  # Red (FP) (pred only)
            2: [0, 0, 1],  # Blue (FN) (double_mask only)
            3: [0, 1, 0],  # Green (TP)
        }
    # Create an empty RGB image
    depth, height, width = image.shape[0], image.shape[1], image.shape[2]
    colored_image = np.zeros((depth, height, width, 3), dtype=np.float32) 
    
    # Map the pixel values to the corresponding colors
    for value, color in color_map.items():
        colored_image[image == value] = color
    return colored_image

def assemble_predictions(images_dir, preds_dir, gt_dir, start_s=0, start_y=0, start_x=0, end_s=6, end_y=8192, end_x=9216):
    tile_depth=3
    tile_width=512
    tile_height=512
    total_slices = ((end_s//tile_depth) * ((end_y-start_y)//tile_height )* ((end_x-start_x)//tile_width))
    slice_num = 0
    print(total_slices, "total slices")
    for s in range(start_s, end_s, 3):
        s_acc_img, s_acc_pred, s_acc_gt = [], [], []
        for y in range(start_y, end_y, 512):
            y_acc_img, y_acc_pred, y_acc_gt = [], [], []
            for x in range(start_x, end_x, 512):
                print(f"Processing volume {s,y,x} | Progress:{slice_num+1}/{total_slices} {(slice_num)/total_slices}", end="\r")
                suffix = r"z{}_y{}_x{}".format(s, y, x)
                
                # load img
                try:
                    img_vol = np.load(os.path.join(images_dir, f"{suffix}.npy"))
                except:
                    img_vol = np.zeros((3, 512,512))
                    print("no img")
                d, h, w = img_vol.shape
                if (d < tile_depth) or (h < tile_height) or (w < tile_width):
                    print("cropping since imgvol shape:", img_vol.shape)
                    img_vol = tio.CropOrPad((tile_depth, tile_height, tile_width))(torch.tensor(img_vol).unsqueeze(0))
                
                # load gt
                try:
                    gt_vol = np.load(os.path.join(gt_dir, f"{suffix}.npy"))
                except:
                    gt_vol = np.zeros((3,512,512))
                    print("no gt")
                if (d < tile_depth) or (h < tile_height) or (w< tile_width):
                    gt_vol = tio.CropOrPad((tile_depth, tile_height, tile_width))(gt_vol)
                
                # load pred
                try:
                    pred_vol = np.load(os.path.join(preds_dir, f"{suffix}.npy"))
                    pred_vol = np.argmax(pred_vol[0], 0) 
                except:
                    print("no pred vol")
                    pred_vol = np.zeros((3, 512,512))
                if (d < tile_depth) or (h < tile_height) or (w < tile_width):
                    pred_vol = tio.CropOrPad((tile_depth, tile_height, tile_width))(pred_vol)
                    
                small_3d_img = []
                small_3d_pred = []
                small_3d_gt = []
                for k in range(3):
                    img = img_vol[k]
                    gt = gt_vol[k]
                    pred = pred_vol[k]
                    small_3d_img += [img]
                    small_3d_gt += [gt]
                    small_3d_pred += [pred]
                    
                small_3d_pred = np.array(small_3d_pred) # (tile depth, tile height, tile width)
                small_3d_gt = np.array(small_3d_gt)
                small_3d_img = np.array(small_3d_img)
                    
                y_acc_gt += [small_3d_gt]
                y_acc_img += [small_3d_img]
                y_acc_pred += [small_3d_pred]
                slice_num+=1
            print(f"Processing volume {s,y,x} | Progress:{slice_num+1}/{total_slices} {(slice_num)/total_slices}")
            s_acc_img += [np.concatenate(y_acc_img, axis=2)]
            s_acc_pred += [np.concatenate(y_acc_pred, axis=2)]
            s_acc_gt += [np.concatenate(y_acc_gt, axis=2)]

        new_img = np.concatenate(s_acc_img, axis=1)
        new_pred = np.concatenate(s_acc_pred, axis=1)
        new_gt = np.concatenate(s_acc_gt, axis=1)
        
        return new_img, new_pred, new_gt
    

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