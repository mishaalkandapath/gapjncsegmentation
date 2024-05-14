#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet_512_5
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
EPOCHS=50
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.001
DATA_DIR=/home/hluo/scratch/data/small_data_3d
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/results
MODEL_NAME="model_512_5"
python /home/hluo/gapjncsegmentation/unet_3d.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --results_dir $RESULTS_DIR
