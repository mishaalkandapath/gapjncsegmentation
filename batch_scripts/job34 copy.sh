#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job34
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
EPOCHS=300
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.001
DATA_DIR=/home/hluo/scratch/select_data_512
MODEL_DIR=/home/hluo/scratch/models
MODEL_NAME="model_job34"
AUGMENT=True
FALSE_NEGATIVE_WEIGHT=None
FALSE_POSITIVE_WEIGHT=None
GAMMA=2
NUM_PREDICTIONS_TO_LOG=10
python /home/hluo/gapjncsegmentation/unet_3d_tversky.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --alpha $FALSE_NEGATIVE_WEIGHT --beta $FALSE_POSITIVE_WEIGHT --gamma $GAMMA --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG