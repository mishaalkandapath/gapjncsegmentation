#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job180b
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=21:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
SEED=999
LOSS_TYPE=focalt
ALPHA=0.002
BETA=0.998
GAMMA=2
EPOCHS=800
BATCH_SIZE=2
NUM_WORKERS=2
LR=0.0000001
NUM_PREDICTIONS_TO_LOG=10
AUGMENT=True
LOAD_MODEL_NAME=model_job136b
LOAD_EPOCH=75
LOAD_MODEL_PATH=/home/hluo/scratch/models/${LOAD_MODEL_NAME}/${LOAD_MODEL_NAME}_epoch_${LOAD_EPOCH}.pth
MODEL_DIR=/home/hluo/scratch/models
RESULTS_DIR=/home/hluo/scratch/model_results
MODEL_NAME="model_job180b"
python ~/gapjncsegmentation/unet_comboloss_no_wandb.py \
  --img_dir_list /home/hluo/scratch/filtered_0_50_3x512x512/original/train /home/hluo/scratch/filtered_100_110_3x512x512_40/original/train \
  --gt_dir_list /home/hluo/scratch/filtered_0_50_3x512x512/ground_truth/train /home/hluo/scratch/filtered_100_110_3x512x512_40/ground_truth/train \
  --valid_img_dir_list /home/hluo/scratch/filtered_0_50_3x512x512/original/valid /home/hluo/scratch/filtered_100_110_3x512x512_40/original/valid \
  --valid_gt_dir_list /home/hluo/scratch/filtered_0_50_3x512x512/ground_truth/valid /home/hluo/scratch/filtered_100_110_3x512x512_40/ground_truth/valid \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --model_name $MODEL_NAME \
  --num_workers $NUM_WORKERS \
  --model_dir $MODEL_DIR \
  --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG \
  --results_dir $RESULTS_DIR \
  --load_model_path $LOAD_MODEL_PATH \
  --augment $AUGMENT \
  --alpha $ALPHA \
  --beta $BETA \
  --gamma $GAMMA \
  --loss_type $LOSS_TYPE \
  --seed $SEED