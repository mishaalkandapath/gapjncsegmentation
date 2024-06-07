#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=getpreds105
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
X_DIR=/home/hluo/scratch/100_110_3x512x512/original
Y_DIR=/home/hluo/scratch/100_110_3x512x512/ground_truth
SAVE_DIR=/home/hluo/scratch/getpreds105
MODEL_NAME=model_job105
EPOCH=84
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
BATCH_SIZE=1
NUM_WORKERS=1
SAVE_VIS=true
SAVE2D=false
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
python /home/hluo/gapjncsegmentation/getpreds.py --x_dir $X_DIR --y_dir $Y_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --save_vis $SAVE_VIS --save2d $SAVE2D --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH