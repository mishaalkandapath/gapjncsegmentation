""" 
remove_invalid_samples.py
Given a directory of subvolumes, this script will remove any subvolumes that do not have the correct shape.

Sample usage:
DATA_DIR="data/tiniest_data_64"
DEPTH=64
HEIGHT=64
WIDTH=64
python remove_invalid_samples.py --data_dir $DATA_DIR --depth $DEPTH --height $HEIGHT --width $WIDTH
"""
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Train a 3D U-Net model on the tiniest dataset")
parser.add_argument("--data_dir", type=str, default="data/tiniest_data_64", help="Directory containing the tiniest dataset")
parser.add_argument("--depth", type=int, help="Directory containing the tiniest dataset")
parser.add_argument("--height", type=int, help="Directory containing the tiniest dataset")
parser.add_argument("--width", type=int, help="Directory containing the tiniest dataset")
parser.add_argument("--print",type=int, default=0, help="Directory containing the tiniest dataset")
args = parser.parse_args()
data_dir = args.data_dir
depth, height, width = args.depth, args.height, args.width
all_paths = os.listdir(data_dir)
num_paths = len(all_paths)
num_removed = 0
for i in range(num_paths):
    print(f"Processed {i}/{num_paths} {((i*100)/num_paths):.4f} (Removed {num_removed})")
    fname = all_paths[i]
    fp = os.path.join(data_dir, fname)
    img = np.load(fp)
    if args.print > 0:
        print(img.shape)
    else:
        if (img.shape[0] != depth) or (img.shape[1] != height) or (img.shape[2] != width):
            print("Removing ", fp, " img shape invalid: ", img.shape)
            os.remove(fp)
            num_removed += 1