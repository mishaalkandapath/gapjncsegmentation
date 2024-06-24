#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job200f
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_NAME="model_job200f"
SEED=29
LOSS_TYPE=focalt
ALPHA=0.3
BETA=0.7
GAMMA=1.5
EPOCHS=800
BATCH_SIZE=4
NUM_WORKERS=2
LR=0.00001
NUM_PREDICTIONS_TO_LOG=10
DEPTH=3
WIDTH=256
HEIGHT=256
DOWNSAMPLE_FACTOR=2
AUGMENT=True
LOAD_MODEL_NAME=model_job111
LOAD_EPOCH=49
LOAD_MODEL_PATH=/home/hluo/scratch/models/${LOAD_MODEL_NAME}/${LOAD_MODEL_NAME}_epoch_${LOAD_EPOCH}.pth
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
LOSS_DIR=/home/hluo/scratch/losses
TRAIN_X_DIRS="/home/hluo/scratch/filtered_0_50_3x512x512/original/train /home/hluo/scratch/filtered_100_110_3x512x512_40/original/train"
TRAIN_Y_DIRS="/home/hluo/scratch/filtered_0_50_3x512x512/ground_truth/train /home/hluo/scratch/filtered_100_110_3x512x512_40/ground_truth/train"
VALID_X_DIRS="/home/hluo/scratch/filtered_0_50_3x512x512/original/valid /home/hluo/scratch/filtered_100_110_3x512x512_40/original/valid"
VALID_Y_DIRS="/home/hluo/scratch/filtered_0_50_3x512x512/ground_truth/valid /home/hluo/scratch/filtered_100_110_3x512x512_40/ground_truth/valid"
python ~/gapjncsegmentation/train.py \
    --downsample_factor $DOWNSAMPLE_FACTOR \
    --train_x_dirs $TRAIN_X_DIRS \
    --train_y_dirs $TRAIN_Y_DIRS \
    --valid_x_dirs $VALID_X_DIRS \
    --valid_y_dirs $VALID_Y_DIRS \
    --model_dir "$MODEL_DIR" \
    --loss_dir "$LOSS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --load_model_path "$LOAD_MODEL_PATH" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --model_name $MODEL_NAME \
    --num_workers $NUM_WORKERS \
    --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG \
    --augment $AUGMENT \
    --alpha $ALPHA \
    --beta $BETA \
    --gamma $GAMMA \
    --loss_type $LOSS_TYPE \
    --seed $SEED \
    --height $HEIGHT \
    --width $WIDTH \
    --depth $DEPTH