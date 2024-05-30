#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=test2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=2:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
START_Z=100
ENDING_DEPTH=104
START_Y=2000
ENDING_HEIGHT=2600
START_X=2000
ENDING_WIDTH=2600
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=128
SUBVOL_WIDTH=128
SAVE_DIR=/Volumes/LaCie/savepred1
IMG_DIR=/Volumes/LaCie/sem_dauer_2_em
MASK_DIR=/Volumes/LaCie/sem_dauer_2_gj_gt
MODEL_NAME=model_job84
EPOCH=49
MODEL_PATH=/Volumes/LaCie/models/${MODEL_NAME}_epoch_${EPOCH}.pth
USE_FULL_VOLUME=False
python /Users/huayinluo/Documents/code/gapjncsegmentation/save3d_large.py --model_path $MODEL_PATH --img_dir $IMG_DIR --mask_dir $MASK_DIR --start_x $START_X --start_y $START_Y --start_z $START_Z --ending_depth $ENDING_DEPTH --ending_height $ENDING_HEIGHT --ending_width $ENDING_WIDTH --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --save_dir $SAVE_DIR --use_full_volume $USE_FULL_VOLUME