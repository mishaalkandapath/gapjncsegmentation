""" 
stitchpreds.py
Given a directory of subvolumes, this script will stitch the predictions together.

Sample usage:
PRED_DIR="/Users/huayinluo/Documents/binarypreds"
SAVE_DIR="/Users/huayinluo/Documents/stitchedpreds"
USE_LINES=false
SHOW_IMG=false
START_S=0
END_S=6
START_X=0
START_Y=0
END_X=9216
END_Y=8192
python stitch.py --pred_dir $PRED_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES --start_s $START_S --end_s $END_S --start_x $START_X --end_x $END_X --start_y $START_Y --end_y $END_Y


# 100-110
SLICES="111_120"
MODELNAME="job202"
EPOCH="325"
PRED_DIR="/home/hluo/scratch/preds/${SLICES}_model_${MODELNAME}_epoch_${EPOCH}_binary"
GT_DIR="/home/hluo/scratch/data/${SLICES}_3x512x512/ground_truth"
IMG_DIR="/home/hluo/scratch/data/${SLICES}_3x512x512/original"
SAVE_DIR="/home/hluo/scratch/stitchedpreds/${SLICES}_model_${MODELNAME}_epoch_${EPOCH}"
USE_LINES=false
SHOW_IMG=false
STITCH2d=true
START_S=0
END_S=4
START_X=0
START_Y=0
END_X=9216
END_Y=8192
python ~/gapjncsegmentation/stitch.py --stitch2d $STITCH2d --pred_dir $PRED_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES --start_s $START_S --end_s $END_S --start_x $START_X --end_x $END_X --start_y $START_Y --end_y $END_Y


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utilities.utilities import get_colored_image, assemble_predictions, assemble_one_slice

    
parser = argparse.ArgumentParser(description="Get evaluation metrics for the model")
parser.add_argument("--pred_dir", type=str, required=True, help="Path to the model file")
parser.add_argument("--gt_dir", type=str, default=None, help="Path to the model file")
parser.add_argument("--img_dir", type=str, default=None, help="Path to the model file")
parser.add_argument("--save_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--use_lines", type=lambda x: (str(x).lower() == 'true'), default=False, help="Path to the results directory")
parser.add_argument("--show_img", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether this is train, test, or valid folder")
parser.add_argument("--stitch2d", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether this is train, test, or valid folder")
parser.add_argument("--alpha", type=float, default=0.4, help="Whether this is train, test, or valid folder")
parser.add_argument("--line_width", type=int, default=3, help="Whether this is train, test, or valid folder")
parser.add_argument("--start_s", type=int, default=0, help="Whether this is train, test, or valid folder")
parser.add_argument("--end_s", type=int, default=1, help="Whether this is train, test, or valid folder")
parser.add_argument("--start_x", type=int, default=0, help="Whether this is train, test, or valid folder")
parser.add_argument("--end_x", type=int, default=1, help="Whether this is train, test, or valid folder")
parser.add_argument("--start_y", type=int, default=0, help="Whether this is train, test, or valid folder")
parser.add_argument("--end_y", type=int, default=1, help="Whether this is train, test, or valid folder")
parser.add_argument("--filename_regex_prefix", type=str, default=None, help="Path to the data directory")
parser.add_argument("--filename_regex_middle", type=str, default=None, help="Path to the data directory")
parser.add_argument("--filename_regex_suffix", type=str, default=None, help="Path to the data directory")
args = parser.parse_args()

print("starting script")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# constants
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
img_dir=args.img_dir
gt_dir=args.gt_dir
pred_dir=args.pred_dir
use_lines = args.use_lines
show_img = args.show_img
alpha = args.alpha
line_width = args.line_width
yellow = [255, 255, 0]

start_s = args.start_s
end_s = args.end_s
start_x = args.start_x
end_x = args.end_x
start_y = args.start_y
end_y = args.end_y

img_ = os.listdir(pred_dir)

# Just assemble the predictions
if args.stitch2d:
    print("=================Just stitching preds=================")
    for s in range(start_s, end_s, 3):
        num_slices = (s%3) if (s%3 > 0) else 3
        for s_num in range(num_slices):
            if args.filename_regex_prefix is None:
                filename_regex = "z"+str(s)+"_y{}_x{}_"+str(s_num)+".png"
            else:
                filename_regex = args.filename_regex_prefix + str(s) + "_y{}_x{}_" + args.filename_regex_middle + str(s_num) + args.filename_regex_suffix
            print(f"---------------------------Started slice {s+s_num}/{end_s} ---------------------------")
            new_pred = assemble_one_slice(pred_dir, filename_regex=filename_regex, start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
            # if use_lines, draw horizontal and vertical lines every 512 pixels for better visualization
            # Note: new_pred is of shape (height, width) and is in grayscale
            print("drawing lines...")
            if use_lines:
                # convert new_pred to RGB
                new_pred = plt.cm.gray(new_pred)
                new_pred = (new_pred * 255).astype("uint8")
                new_pred = new_pred[:, :, :3]
                # draw horizontal lines
                for i in range(0, new_pred.shape[0], 512):
                    new_pred[i, :] = yellow
                # draw vertical lines
                for i in range(0, new_pred.shape[1], 512):
                    new_pred[:, i] = yellow
        
            plt.imsave(os.path.join(save_dir, f"slice{s+s_num}.png"), new_pred, cmap="gray")
            print(f"---------------------------Finished slice {s+s_num}---------------------------")
    exit(0)

# Compare with gt and assemble tp, fp, tn, fn
else:    
    print("=================Stitching preds with gt and img=================")
    # stich together preds
    new_img, new_pred, new_gt = assemble_predictions(img_dir, pred_dir, gt_dir)
    volume_depth = new_img.shape[0]
    print("shape", new_img.shape, new_pred.shape, new_gt.shape)
    unique_gt_labels, unique_gt_counts = np.unique(new_gt, return_counts=True)
    print("unique gt:", unique_gt_labels)
    print("unique gt counts:", unique_gt_counts)
    height, width = new_img.shape[1], new_img.shape[2]


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

        