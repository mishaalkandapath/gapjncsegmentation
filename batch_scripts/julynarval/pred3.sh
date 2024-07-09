#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=pred3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_NAME=model_job403
EPOCH=59
SLICES="150_180"
CROP_SIZE=512
STRIDE=256
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
X_DIR=/home/hluo/scratch/data/${SLICES}_3x${CROP_SIZE}x${CROP_SIZE}_stride${STRIDE}/original
SAVE_DIR=/home/hluo/scratch/preds/${SLICES}_${MODEL_NAME}_epoch_${EPOCH}
SAVE2D=true
SAVECOMB=false
USEALLSUBFOLDERS=false
PRED_MEMB=false
BATCH_SIZE=1
NUM_WORKERS=4
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
python /home/hluo/gapjncsegmentation/getpreds.py --pred_memb $PRED_MEMB --useallsubfolders $USEALLSUBFOLDERS --x_dir $X_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --save2d $SAVE2D --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --savecomb $SAVECOMB