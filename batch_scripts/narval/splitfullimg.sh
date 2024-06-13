#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=savepred113
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
START_Z=111
ENDING_DEPTH=120
START_Y=0
START_X=0
USE_FULL_VOLUME=True
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
SAVE_DIR=/home/hluo/scratch/111_120_3x512x512
IMG_DIR=/home/hluo/scratch/111_120_fullimgs/original
MASK_DIR=/home/hluo/scratch/111_120_fullimgs/ground_truth
STEP_Z=3
STEP_Y=512
STEP_X=512
python /home/hluo/gapjncsegmentation/split_full_img.py --step_z $STEP_Z --step_x $STEP_X --step_y $STEP_Y --img_dir $IMG_DIR --mask_dir $MASK_DIR --start_x $START_X --start_y $START_Y --start_z $START_Z --ending_depth $ENDING_DEPTH --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --save_dir $SAVE_DIR --use_full_volume $USE_FULL_VOLUME
