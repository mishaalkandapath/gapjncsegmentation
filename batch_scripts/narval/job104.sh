#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job104
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
EPOCHS=100
BATCH_SIZE=4
NUM_WORKERS=4
LR=0.00002
LOAD_MODEL_PATH=/home/hluo/scratch/models/model_job102/model_job102_epoch_48.pth
DATA_DIR=/home/hluo/scratch/select_dauer_data_512
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
MODEL_NAME="model_job104"
ALPHA=0.986
GAMMA=3
NUM_PREDICTIONS_TO_LOG=10
AUGMENT=True
python /home/hluo/gapjncsegmentation/unet_3d_no_wandb.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --alpha $ALPHA --gamma $GAMMA --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG --results_dir $RESULTS_DIR --augment $AUGMENT --load_model_path $LOAD_MODEL_PATH
