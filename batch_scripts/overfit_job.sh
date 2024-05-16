#!/bin/sh
EPOCHS=10
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.001
DATA_DIR=/home/hluo/scratch/one_512
MODEL_DIR=/home/hluo/scratch/models
MODEL_NAME="overfit_one"
AUGMENT=False
ALPHA=0.95
GAMMA=3
NUM_PREDICTIONS_TO_LOG=10
python unet_3d.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --alpha $ALPHA --gamma $GAMMA --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG --loss_type $LOSS_TYPE
