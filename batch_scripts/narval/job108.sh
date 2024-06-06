#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job108
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
LOSS_TYPE=combo
ALPHA=0.8
CE_RATIO=0.5
EPOCHS=50
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.0001
NUM_PREDICTIONS_TO_LOG=10
AUGMENT=True
LOAD_MODEL_PATH=/home/hluo/scratch/models/model_job89/model_job89_epoch_49.pth
DATA_DIR=/home/hluo/scratch/select_dauer_data_512
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
MODEL_NAME="model_job108"
python ~/gapjncsegmentation/unet_comboloss_no_wandb.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG --results_dir $RESULTS_DIR --augment $AUGMENT --load_model_path $LOAD_MODEL_PATH --alpha $ALPHA --ce_ratio $CE_RATIO --loss_type $LOSS_TYPE
