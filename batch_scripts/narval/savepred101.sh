#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=savepred101
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
START_Z=100
ENDING_DEPTH=111
START_Y=0
ENDING_HEIGHT=2600
START_X=0
ENDING_WIDTH=2600
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
SAVE_DIR=/home/hluo/scratch/results/savepred101
IMG_DIR=/home/hluo/scratch/sem_dauer_2_em
MASK_DIR=/home/hluo/scratch/sem_dauer_2_gj_gt
MODEL_NAME=model_job101
EPOCH=31
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
USE_FULL_VOLUME=True
python /home/hluo/gapjncsegmentation/save3d_large.py --model_path $MODEL_PATH --img_dir $IMG_DIR --mask_dir $MASK_DIR --start_x $START_X --start_y $START_Y --start_z $START_Z --ending_depth $ENDING_DEPTH --ending_height $ENDING_HEIGHT --ending_width $ENDING_WIDTH --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --save_dir $SAVE_DIR --use_full_volume $USE_FULL_VOLUME
