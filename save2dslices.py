"""
save2dslices.py
Given directories of images and masks, this script will filter for 2D images and masks where gap junctions are present and perform train/test split

Sample usage:


"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utilities.utilities import get_z_y_x

parser = argparse.ArgumentParser(description="save 2D slices")
parser.add_argument('--img_dir', type=str, required=True, help='Directory containing the images')
parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing the masks')
parser.add_argument('--save_dir', type=str, default="", help='Save directory')
args = parser.parse_args()


img_dir = args.img_dir
mask_dir = args.mask_dir
img_files = os.listdir(img_dir)
mask_files = os.listdir(mask_dir)

# create directories
if not os.path.exists("small_data"):
    os.makedirs("small_data")
if not os.path.exists("small_data/original"):
    os.makedirs("small_data/original")
if not os.path.exists("small_data/ground_truth"):
    os.makedirs("small_data/ground_truth")
if not os.path.exists("small_data/original/train"):
    os.makedirs("small_data/original/train")
if not os.path.exists("small_data/original/valid"):
    os.makedirs("small_data/original/valid")
if not os.path.exists("small_data/original/test"):
    os.makedirs("small_data/original/test")
if not os.path.exists("small_data/ground_truth/train"):
    os.makedirs("small_data/ground_truth/train")
if not os.path.exists("small_data/ground_truth/valid"):
    os.makedirs("small_data/ground_truth/valid")
if not os.path.exists("small_data/ground_truth/test"):
    os.makedirs("small_data/ground_truth/test")
 
mask_pattern = r"SEM_adult_gj_segmentation_WL.vsseg_export_s(\d+)_Y(\d+)_X(\d+).png"
img_pattern = r"Dataset8_export_s(\d+)_Y(\d+)_X(\d+).png"
# choose images with gap junctions
# 0, 2: background
# 1, 3: gap junction
imgs = []
masks = []
unique_values = []
background_values = [0, 2]
gap_junction_values = [1, 3]
for i in range(len(img_files)):
    img_f = img_files[i]
    mask_f = mask_files[i]
    mask = cv2.imread(os.path.join(mask_dir, mask_f))
    img = cv2.imread(os.path.join(img_dir, img_f))
    processed_mask = mask.copy()
    processed_mask[np.isin(processed_mask, background_values)] = 0
    processed_mask[np.isin(processed_mask, gap_junction_values)] = 1
    
    # save imgs where gap junctions are present
    if (len(np.unique(processed_mask))>1):
        labels = np.unique(processed_mask)
        for label in labels:
            if label not in unique_values:
                unique_values.append(label)
                print(f"Unique value {label} found in mask {mask_f}")
        mask[mask == 3] = 1
        mask[mask == 2] = 0
        mask_visual = mask.copy()
        mask_visual[mask_visual == 1] = 255
        print(f"Unique values in mask {mask_f}: {np.unique(mask)}")
        img_z, img_y, img_x = get_z_y_x(img_f, img_pattern)
        mask_z, mask_y, mask_x = get_z_y_x(mask_f, mask_pattern)
        if (img_z != mask_z) or (img_y != mask_y) or (img_x != mask_x):
            print(f"Image and mask coordinates do not match: {img_f}, {mask_f}")
            print(f"Image: {img_z}, {img_y}, {img_x}")
            print(f"Mask: {mask_z}, {mask_y}, {mask_x}")
            continue
        plt.imsave(f"small_data/original/z{img_z}_y{img_y}_x{img_x}.png", img)
        plt.imsave(f"small_data/ground_truth/z{mask_z}_y{mask_y}_x{mask_x}.png", mask)
                
    print(f"Checked {i} out of {len(img_files)} | progress: {i/len(img_files)*100:.2f}%")
# split data into train, valid, test
num_imgs = len(imgs)
indices = np.arange(num_imgs)
np.random.shuffle(indices)
train_indices = indices[:int(0.6*num_imgs)]
valid_indices = indices[int(0.6*num_imgs):int(0.8*num_imgs)]
test_indices = indices[int(0.8*num_imgs):]
imgs = np.array(imgs)
masks = np.array(masks)

train_imgs = imgs[train_indices]
valid_imgs = imgs[valid_indices]
test_imgs = imgs[test_indices]

train_masks = masks[train_indices]
valid_masks = masks[valid_indices]
test_masks = masks[test_indices]
print(f"Train: {train_masks.shape}, {train_imgs.shape}") # (num_imgs, 512, 512, 3)
# sanity check: do images and masks match?
i=1
tmp_img = train_imgs[i]
tmp_mask = train_masks[i]
tmp_mask[tmp_mask > 0] = 255
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(tmp_img)
ax[1].imshow(tmp_mask)
# save imgs
for i,img in enumerate(train_imgs):
    cv2.imwrite(f"small_data/original/train/{i}.png", train_imgs[i])
    print(f"Saved train {i}")

for i,img in enumerate(valid_imgs):
    cv2.imwrite(f"small_data/original/valid/{i}.png", valid_imgs[i])
    print(f"Saved valid {i}")

for i,img in enumerate(test_imgs):
    cv2.imwrite(f"small_data/original/test/{i}.png", test_imgs[i])
    print(f"Saved test {i}")

# save masks
for i,img in enumerate(train_masks):
    plt.imsave(f"small_data/ground_truth/train/{i}.png", train_masks[i][:, :, 1])
    print(f"Saved train mask {i}")

for i,img in enumerate(valid_masks):
    plt.imsave(f"small_data/ground_truth/valid/{i}.png", valid_masks[i][:, :, 1])
    print(f"Saved valid mask {i}")

for i,img in enumerate(test_masks):
    plt.imsave(f"small_data/ground_truth/test/{i}.png", test_masks[i][:, :, 1])
    print(f"Saved test mask {i}")