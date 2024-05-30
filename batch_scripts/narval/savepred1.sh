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
START_X=0
START_Y=0
START_Z=100
ENDING_DEPTH=104
ENDING_HEIGHT=500
ENDING_WIDTH=500
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=100
SUBVOL_WIDTH=100
SAVE_DIR=/home/hluo/scratch/results/savepred1
IMG_DIR=/home/hluo/scratch/sem_dauer_2_em
MASK_DIR=/home/hluo/scratch/sem_dauer_2_gj_gt
MODEL_NAME=model_job84
EPOCH=49
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
python ~/gapjncsegmentation/save3d_large.py --model_path $MODEL_PATH --img_dir $IMG_DIR --mask_dir $MASK_DIR --start_x $START_X --start_y $START_Y --start_z $START_Z --ending_depth $ENDING_DEPTH --ending_height $ENDING_HEIGHT --ending_width $ENDING_WIDTH --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --save_dir $SAVE_DIR