#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job142
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
SEED=29
LOSS_TYPE=focalt
ALPHA=0.4
BETA=0.6
GAMMA=2
EPOCHS=800
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.00001
NUM_PREDICTIONS_TO_LOG=10
AUGMENT=True
LOAD_MODEL_NAME=model_job111
LOAD_EPOCH=49
LOAD_MODEL_PATH=/home/hluo/scratch/models/${LOAD_MODEL_NAME}/${LOAD_MODEL_NAME}_epoch_${LOAD_EPOCH}.pth
DATA_DIR=/home/hluo/scratch/filtered_0_50_3x512x512
CELLMASK_DIR=/home/hluo/scratch/filtered_0_50_3x512x512/cell_mask
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
MODEL_NAME="model_job142"
python ~/gapjncsegmentation/unet_comboloss_no_wandb.py --cellmask_dir $CELLMASK_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG --results_dir $RESULTS_DIR --augment $AUGMENT --load_model_path $LOAD_MODEL_PATH --alpha $ALPHA --beta $BETA --gamma $GAMMA --loss_type $LOSS_TYPE --seed $SEED
