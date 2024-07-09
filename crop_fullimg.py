""" 

Sample Usage: (multiline)

IMG_DIR="/Volumes/LaCie 1/train050/test_img"
GT_DIR="/Volumes/LaCie 1/train050/test_gt"
SAVE_IMG_DIR="/Volumes/LaCie 1/train050/test_non-overlap/imgs2"
SAVE_GT_DIR="/Volumes/LaCie 1/train050/test_non-overlap/gt2"
SAVE_VIS_DIR="/Volumes/LaCie 1/train050/test_non-overlap/vis2"
CROP_SIZE=1024
STRIDE=1024
DEPTH=2
GT_PROP=0.0000001


SLICES="0_50"



SLICES="111_120"


module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv

SLICES="111_120"
CROP_SIZE=256
STRIDE=128
DEPTH=3
GT_PROP=0.0000001
IMG_DIR="/home/hluo/scratch/data/${SLICES}_fullimgs/original"
GT_DIR="/home/hluo/scratch/data/${SLICES}_fullimgs/ground_truth"
SAVE_IMG_DIR="/home/hluo/scratch/data/${SLICES}_${DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${STRIDE}/original"
SAVE_GT_DIR="/home/hluo/scratch/data/${SLICES}_${DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${STRIDE}/ground_truth"
SAVE_VIS_DIR="/home/hluo/scratch/data/0_50_${DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${STRIDE}/vis"
python ~/gapjncsegmentation/crop_fullimg.py --img_dir $IMG_DIR \
                          --gt_dir $GT_DIR \
                          --save_img_dir $SAVE_IMG_DIR \
                          --save_gt_dir $SAVE_GT_DIR \
                          --save_vis_dir $SAVE_VIS_DIR \
                          --gt_proportion $GT_PROP \
                          --save_gt_255 \
                          --stride $STRIDE \
                          --crop_size $CROP_SIZE \
                          --suffix png \
                          --depth $DEPTH
"""

import argparse
import os
from utilities.utilities_train import generate_cropped_3d_dataset, generate_cropped_3d_dataset_img_only

parser = argparse.ArgumentParser(description='Crop and save images and ground truth data.')
parser.add_argument('--img_dir', type=str, help='Directory containing input images')
parser.add_argument('--gt_dir', type=str, help='Directory containing ground truth data')
parser.add_argument('--save_img_dir', type=str, help='Directory to save cropped images')
parser.add_argument('--save_gt_dir', type=str, default=None, help='Directory to save cropped ground truth data')
parser.add_argument('--save_vis_dir', type=str, default=None, help='Directory to save visualizations')
parser.add_argument('--gt_proportion', type=float, default=0.0000001, help='Proportion of ground truth data to save')
parser.add_argument('--save_gt_255', action='store_true', help='Save ground truth data as 255 values')
parser.add_argument('--stride', type=int, default=512, help='Stride for cropping')
parser.add_argument('--crop_size', type=int, default=512, help='Size of cropped images')
parser.add_argument('--suffix', type=str, default='png', help='Suffix for saved images')
parser.add_argument('--depth', type=int, help='Depth of directory structure for saving')

args = parser.parse_args()

img_dir = args.img_dir
gt_dir = args.gt_dir
save_img_dir = args.save_img_dir
save_gt_dir = args.save_gt_dir
save_vis_dir = args.save_vis_dir
gt_proportion = args.gt_proportion
save_gt_255 = args.save_gt_255
stride = args.stride
crop_size = args.crop_size
suffix = args.suffix
depth = args.depth

if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)
if save_vis_dir is not None:
    if not os.path.exists(save_vis_dir):
        os.makedirs(save_vis_dir)
if save_gt_dir is not None:
    if not os.path.exists(save_gt_dir):
        os.makedirs(save_gt_dir)


if gt_dir is None:
    print("Cropping img dir only")
    generate_cropped_3d_dataset_img_only(img_dir, save_img_dir, save_vis_dir=save_vis_dir, crop_size=crop_size, stride=stride, suffix=suffix)
else:
    print("Cropping img and gt dir")
    generate_cropped_3d_dataset(img_dir, gt_dir, save_img_dir, save_gt_dir, save_vis_dir=save_vis_dir, save_depth=depth, crop_size=crop_size, stride=stride, gt_proportion=gt_proportion, suffix=suffix, save_gt_255=save_gt_255)