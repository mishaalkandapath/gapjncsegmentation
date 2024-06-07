#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=pred111
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
DATA_DIR=/home/hluo/scratch/100_110_3x512x512
SAVE_DIR=/home/hluo/scratch/preds/pred111
MODEL_NAME=model_job111
EPOCH=49
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
python  /home/hluo/gapjncsegmentation/pred_on_subvolumes.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH
