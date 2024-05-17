#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=test2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=2:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_PATH=/home/hluo/scratch/models/model_job20/model_job20_epoch_99.pth
DATA_DIR=/home/hluo/scratch/small_data_256
RESULTS_DIR=/home/hluo/scratch/results/test2a
python test.py --model_path $MODEL_PATH --data_dir $DATA_DIR --results_dir $RESULTS_DIR