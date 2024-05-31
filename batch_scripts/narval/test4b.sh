#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=test4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_NAME=model_job93
EPOCH=49
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
DATA_DIR=/home/hluo/scratch/select_dauer_data_512
RESULTS_DIR=/home/hluo/scratch/results/test4
python ~/gapjncsegmentation/test.py --model_path $MODEL_PATH --data_dir $DATA_DIR --results_dir $RESULTS_DIR