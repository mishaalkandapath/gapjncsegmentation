#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet_short
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=3:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
EPOCHS=20
BATCH_SIZE=8
LR=0.001
DATA_DIR=/home/hluo/scratch/small_data_256
MODEL_NAME="model_256_1"
python /home/hluo/projects/def-mzhen/hluo/gapjncsegmentation/unet_3d.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME

