""" 
stitchpreds.py
Given a directory of subvolumes, this script will stitch the predictions together.

Sample usage:
DATA_DIR="/Volumes/LaCie/june4/savepred6/"
SAVE_DIR="/Volumes/LaCie/gapjncsave6/"
USE_LINES=false
SHOW_IMG=false
python stitch.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utilities.utilities import get_colored_image, assemble_predictions

    
parser = argparse.ArgumentParser(description="Get evaluation metrics for the model")
parser.add_argument("--save_dir", type=str, required=True, help="Path to the model file")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--use_lines", type=bool, default=False, help="Path to the results directory")
parser.add_argument("--show_img", type=bool, default=False, help="Whether this is train, test, or valid folder")
parser.add_argument("--alpha", type=float, default=0.4, help="Whether this is train, test, or valid folder")
parser.add_argument("--line_width", type=int, default=3, help="Whether this is train, test, or valid folder")
args = parser.parse_args()


# constants
data_dir= args.data_dir
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
img_dir=os.path.join(data_dir, "original")
gt_dir=os.path.join(data_dir, "ground_truth")
pred_dir=os.path.join(data_dir, "pred")
use_lines = args.use_lines
show_img = args.show_img
alpha = args.alpha
line_width = args.line_width
yellow = [255, 255, 0]


# stich together preds
new_img, new_pred, new_gt = assemble_predictions(img_dir, pred_dir, gt_dir)
volume_depth = new_img.shape[0]
print("shape", new_img.shape, new_pred.shape, new_gt.shape)
unique_gt_labels, unique_gt_counts = np.unique(new_gt, return_counts=True)
print("unique gt:", unique_gt_labels)
print("unique gt counts:", unique_gt_counts)


# save with all labels
binary_gt = new_gt.copy()
binary_gt[binary_gt != 0] = 1
combined_volume = np.asarray((binary_gt * 2 + new_pred))
color_combined_volume = get_colored_image(combined_volume)
if show_img:
    for k in range(volume_depth):
        print(f"Saved {k}")
        fig = plt.figure(num=1)
        if use_lines:
            lined_img = new_img[k].copy()
            lined_img = (lined_img - np.min(lined_img))/(np.max(lined_img)-np.min(lined_img))
            lined_img = np.stack((lined_img,) * 3, axis=-1)
            for y in range(0, height, 512):
                lined_img[y:y+line_width, :] = yellow
            for x in range(0, width, 512):
                lined_img[:, x:x+line_width] = yellow
            plt.imshow(lined_img)
        else:
            plt.imshow(new_img[k], cmap='gray')
        plt.imshow(color_combined_volume[k], alpha=alpha)
        plt.savefig(save_dir + f"combwithimg_slice{k}.png", dpi=800)
        plt.close("all")
else:
    for k in range(volume_depth):
        plt.imsave(save_dir + f"comb_slice{k}.png", color_combined_volume[k], cmap="gray")
        print(f"Saved slice {k}")
    

# save by confidence
height, width = new_img.shape[1], new_img.shape[2]
for label in unique_gt_labels:
    print("Saving confidence ", label)
    confidence_gt = new_gt.copy()
    confidence_gt[confidence_gt != label] = 0
    confidence_gt[confidence_gt == label] = 1
    print(np.unique(confidence_gt, return_counts=True))
    combined_volume = np.asarray((confidence_gt * 2 + new_pred))
    color_combined_volume = get_colored_image(combined_volume)
    if show_img:
        for k in range(volume_depth):
            print(f"Saved {k}")
            fig = plt.figure(num=1)
            if use_lines:
                lined_img = new_img[k].copy()
                lined_img = (lined_img - np.min(lined_img))/(np.max(lined_img)-np.min(lined_img))
                lined_img = np.stack((lined_img,) * 3, axis=-1)
                for y in range(0, height, 512):
                    lined_img[y:y+line_width, :] = yellow
                for x in range(0, width, 512):
                    lined_img[:, x:x+line_width] = yellow
                plt.imshow(lined_img)
            else:
                plt.imshow(new_img[k], cmap='gray')
            plt.imshow(color_combined_volume[k], alpha=alpha)
            plt.savefig(save_dir + f"combwithimg_confidence{str(int(label))}_slice{k}.png", dpi=800)
            plt.close("all")
    else:
        for k in range(volume_depth):
            plt.imsave(save_dir + f"comb_confidence{str(int(label))}_slice{k}.png", color_combined_volume[k], cmap="gray")
            print(f"Saved slice {k}")

    