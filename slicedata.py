""" Section one 512 x 512 image into sixteen 128 x 128 tiles."""
import os
import argparse


parser = argparse.ArgumentParser(description="Slice images into smaller images")
args = parser.parse_args()

old_dir = "small_data_3d_5"
new_dir = "small_data_256_overlap"

# directory with the original images (full size)
old_img_dir = os.path.join(old_dir, "original")
old_mask_dir = os.path.join(old_dir, "ground_truth")
# directory to save the new images (small size)
new_img_dir = os.path.join(new_dir, "original")
new_mask_dir = os.path.join(new_dir, "ground_truth")

# Create new directories as needed
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
if not os.path.exists(new_img_dir):
    os.makedirs(new_img_dir)
if not os.path.exists(new_img_dir + "/train"):
    os.makedirs(new_img_dir + "/train")
if not os.path.exists(new_img_dir + "/test"):
    os.makedirs(new_img_dir + "/test")
if not os.path.exists(new_img_dir + "/valid"):
    os.makedirs(new_img_dir + "/valid")
if not os.path.exists(new_mask_dir):
    os.makedirs(new_mask_dir)
if not os.path.exists(new_mask_dir + "/train"):
    os.makedirs(new_mask_dir + "/train")
if not os.path.exists(new_mask_dir + "/test"):
    os.makedirs(new_mask_dir + "/test")
if not os.path.exists(new_mask_dir + "/valid"):
    os.makedirs(new_mask_dir + "/valid")
img_paths_train = os.listdir(old_img_dir + "/train")
mask_paths_train = os.listdir(old_mask_dir + "/train")
img_paths_test = os.listdir(old_img_dir + "/test")
mask_paths_test = os.listdir(old_mask_dir + "/test")
img_paths_valid = os.listdir(old_img_dir + "/valid")
img_paths_valid = os.listdir(old_mask_dir + "/valid")

print(len(img_paths_train), len(mask_paths_train), len(img_paths_test), len(mask_paths_test))

old_height = 512

# CHANGE ME
new_height = 256 # new height of each tile
# stride = 512 // new_height # stride to move the window
stride = 128
print(f"num new imgs: {old_height // stride}")
import numpy as np

# train set
counter = 0 # for naming the new files
num_imgs = len(img_paths_train)
for k in tqdm.tqdm(range(num_imgs)):
    fp = img_paths_train[k]
    img = np.load(old_img_dir + "/train/" + fp)
    mask = np.load(old_mask_dir + "/train/" + fp)
    
    # split the image into tiles
    for i in range(0, old_height, stride):
        for j in range(0, old_height, stride):
            img_tile = img[:, i:i+new_height, j:j+new_height]
            mask_tile = mask[:, i:i+new_height, j:j+new_height]
            np.save(new_img_dir + "/train/" + str(counter) + ".npy", img_tile)
            np.save(new_mask_dir + "/train/" + str(counter) + ".npy", mask_tile)
            counter += 1

# test set
counter = 0
num_imgs = len(img_paths_test)
for k in tqdm.tqdm(range(num_imgs)):
    fp = img_paths_test[k]
    img = np.load(old_img_dir + "/test/" + fp)
    mask = np.load(old_mask_dir + "/test/" + fp)
    
    for i in range(0, old_height, stride):
        for j in range(0, old_height, stride):
            img_tile = img[:, i:i+new_height, j:j+new_height]
            mask_tile = mask[:, i:i+new_height, j:j+new_height]
            np.save(new_img_dir + "/test/" + str(counter) + ".npy", img_tile)
            np.save(new_mask_dir + "/test/" + str(counter) + ".npy", mask_tile)
            counter += 1

# valid set
counter = 0 
num_imgs = len(img_paths_valid)
for k in tqdm.tqdm(range(num_imgs)):
    fp = img_paths_valid[k]
    img = np.load(old_img_dir + "/valid/" + fp)
    mask = np.load(old_mask_dir + "/valid/" + fp)
    
    for i in range(0, old_height, stride):
        for j in range(0, old_height, stride):
            img_tile = img[:, i:i+new_height, j:j+new_height]
            mask_tile = mask[:, i:i+new_height, j:j+new_height]
            np.save(new_img_dir + "/valid/" + str(counter) + ".npy", img_tile)
            np.save(new_mask_dir + "/valid/" + str(counter) + ".npy", mask_tile)
            counter += 1