#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=data1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
SLICES="100_110"
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