#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=stitch3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODELNAME=model_job403
EPOCH=59
SLICES="150_180"
PRED_DIR="/home/hluo/scratch/preds/${SLICES}_${MODELNAME}_epoch_${EPOCH}_binary"
SAVE_DIR="/home/hluo/scratch/stitchedpreds/${SLICES}_${MODELNAME}_epoch_${EPOCH}"
USE_LINES=false
SHOW_IMG=false
STITCH2d=true
START_S=150
END_S=180
START_X=0
START_Y=0
END_X=9216
END_Y=8192
FILENAME_REGEX_PREFIX="SEM_dauer_2_export_s"
FILENAME_REGEX_MIDDLE=".png_"
FILENAME_REGEX_SUFFIX=".png "
python ~/gapjncsegmentation/stitch.py --filename_regex_prefix $FILENAME_REGEX_PREFIX --filename_regex_middle $FILENAME_REGEX_MIDDLE --filename_regex_suffix $FILENAME_REGEX_SUFFIX --stitch2d $STITCH2d --pred_dir $PRED_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES --start_s $START_S --end_s $END_S --start_x $START_X --end_x $END_X --start_y $START_Y --end_y $END_Y