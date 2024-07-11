#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=stitch2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODELNAME=model_job403
EPOCH=39
SLICES="150_180"
PRED_DIR="/home/hluo/scratch/preds/${SLICES}_model_${MODELNAME}_epoch_${EPOCH}_binary"
SAVE_DIR="/home/hluo/scratch/stitchedpreds/${SLICES}_model_${MODELNAME}_epoch_${EPOCH}"
USE_LINES=false
SHOW_IMG=false
STITCH2d=true
START_S=0
END_S=30
START_X=0
START_Y=0
END_X=9216
END_Y=8192
python ~/gapjncsegmentation/stitch.py --stitch2d $STITCH2d --pred_dir $PRED_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES --start_s $START_S --end_s $END_S --start_x $START_X --end_x $END_X --start_y $START_Y --end_y $END_Y