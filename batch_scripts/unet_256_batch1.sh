#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet_256_long
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
EPOCHS=25
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.002
DATA_DIR=/home/hluo/scratch/small_data_256
MODEL_DIR=/home/hluo/scratch/models
MODEL_NAME="model_256_batch1"
python /home/hluo/gapjncsegmentation/unet_3d.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR

