#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job401ccont2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=11:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
echo "Starting job401c: train on 100-120, validate on 0-50 (no downsample + only flip augment)"
MODEL_NAME="model_job401ccont2"
SEED=9
INTERMEDIATE_WEIGHT=0.6
USE2d3d=True
LOSS_TYPE=focalt
ALPHA=0.04
BETA=0.96
GAMMA=1.5
EPOCHS=400
BATCH_SIZE=1
NUM_WORKERS=2
LR=0.0001
NUM_PREDICTIONS_TO_LOG=3
DEPTH=3
WIDTH=512
HEIGHT=512
AUGMENT=False
COLOUR_AUGMENT=False
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
LOSS_DIR=/home/hluo/scratch/losses
VALID_SLICES="111_120"
CROP_SIZE=512
CROP_DEPTH=3
TRAIN_STRIDE=512
VALID_STRIDE=512
TRAIN_X_DIRS="/home/hluo/scratch/data/150_170small_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride256/original /home/hluo/scratch/data/100_110_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${TRAIN_STRIDE}/original /home/hluo/scratch/data/0_50_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${TRAIN_STRIDE}/original"
TRAIN_Y_DIRS="/home/hluo/scratch/data/150_170small_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride256/ground_truth /home/hluo/scratch/data/100_110_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${TRAIN_STRIDE}/ground_truth /home/hluo/scratch/data/0_50_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${TRAIN_STRIDE}/ground_truth"
VALID_X_DIRS="/home/hluo/scratch/data/${VALID_SLICES}_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${VALID_STRIDE}/original"
VALID_Y_DIRS="/home/hluo/scratch/data/${VALID_SLICES}_${CROP_DEPTH}x${CROP_SIZE}x${CROP_SIZE}_stride${VALID_STRIDE}/ground_truth"
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