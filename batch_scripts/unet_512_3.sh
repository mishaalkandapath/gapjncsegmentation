#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet_512_3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
EPOCHS=100
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.001
DATA_DIR=/home/hluo/scratch/small_data_3d_3
MODEL_DIR=/home/hluo/scratch/models
LOAD_MODEL_PATH="model_512_3"
MODEL_NAME="model_512_3_two"
RESULTS_DIR=/home/hluo/scratch/results
python /home/hluo/gapjncsegmentation/unet_3d.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --results_dir $RESULTS_DIR --load_modeel_path $LOAD_MODEL_PATH
