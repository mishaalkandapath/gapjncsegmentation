#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job206b
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_NAME="model_job206b"
SEED=29
INTERMEDIATE_WEIGHT=1.6
USE2d3d=True
LOSS_TYPE=focalt
ALPHA=0.008
BETA=0.992
GAMMA=1.5
EPOCHS=400
BATCH_SIZE=1
NUM_WORKERS=2
LR=0.00001
NUM_PREDICTIONS_TO_LOG=3
DEPTH=3
WIDTH=512
HEIGHT=512
AUGMENT=True
LOAD_MODEL_NAME=model_job204c
LOAD_EPOCH=52
LOAD_MODEL_PATH=/home/hluo/scratch/models/${LOAD_MODEL_NAME}/${LOAD_MODEL_NAME}_epoch_${LOAD_EPOCH}.pth
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
LOSS_DIR=/home/hluo/scratch/losses
TRAIN_X_DIRS="/home/hluo/scratch/0_50_3x512x512_filtered/original/train /home/hluo/scratch/100_110_3x512x512_filtered40/original/train"
TRAIN_Y_DIRS="/home/hluo/scratch/0_50_3x512x512_filtered/ground_truth/train /home/hluo/scratch/100_110_3x512x512_filtered40/ground_truth/train"
VALID_X_DIRS="/home/hluo/scratch/111_120_3x512x512/original"
VALID_Y_DIRS="/home/hluo/scratch/111_120_3x512x512/ground_truth"
python ~/gapjncsegmentation/train.py \
    --intermediate_weight $INTERMEDIATE_WEIGHT \
    --use2d3d $USE2d3d \
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