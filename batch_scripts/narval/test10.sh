#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=test10
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_NAME=model_job92
EPOCH=49
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
DATA_DIR=/home/hluo/scratch/results/savepred5
RESULTS_DIR=/home/hluo/scratch/results/test10
FOLDER_TYPE=train
python ~/gapjncsegmentation/test_savepred.py --model_path $MODEL_PATH --data_dir $DATA_DIR --results_dir $RESULTS_DIR --folder_type $FOLDER_TYPE