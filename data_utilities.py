import os
import re
import cv2
import numpy as np

def get_z_y_x(file_name, pattern):
    """ get z, y, x from file name (uses basename of file_name)"""
    file_name = os.path.basename(file_name)
    match = re.match(pattern, file_name)
    if match:
        if len(match.groups()) != 3:
            return None
        z, y, x = match.groups()
        return int(z), int(y), int(x)
    else:
        return None
    
def get_img_by_coords(z, y, x, img_files, img_pattern, extension=".npy"):
    """ get path of image by z, y, x """
    for i in range(len(img_files)):
        img_file = img_files[i]
        z_, y_, x_ = get_z_y_x(img_file, img_pattern)
        if z == z_ and y == y_ and x == x_:
            if extension == ".npy":
                return np.load(img_file)
            elif extension == ".png":
                return cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    return None

def get_3d_slice(z,y,x, img_files, mask_files, img_pattern, mask_pattern, depth=1, width=512, height=512):
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
        