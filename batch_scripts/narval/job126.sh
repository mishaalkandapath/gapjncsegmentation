#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job126
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
LOSS_TYPE=focalt
ALPHA=0.04
BETA=0.96
GAMMA=2
EPOCHS=100
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.00001
NUM_PREDICTIONS_TO_LOG=10
AUGMENT=True
LOAD_MODEL_NAME=model_job103
LOAD_EPOCH=51
LOAD_MODEL_PATH=/home/hluo/scratch/models/${LOAD_MODEL_NAME}/${LOAD_MODEL_NAME}_epoch_${LOAD_EPOCH}.pth
DATA_DIR=/home/hluo/scratch/filtered_100_110_3x512x512_20
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
MODEL_NAME="model_job126"
python ~/gapjncsegmentation/unet_comboloss_no_wandb.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG --results_dir $RESULTS_DIR --augment $AUGMENT --load_model_path $LOAD_MODEL_PATH --alpha $ALPHA --beta $BETA --gamma $GAMMA --loss_type $LOSS_TYPE
