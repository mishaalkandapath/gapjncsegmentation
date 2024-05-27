#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=test9
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=2:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_PATH=/home/hluo/scratch/models/model_job20f/model_job20f_epoch_139.pth
DATA_DIR=/home/hluo/scratch/select_dauer_data_512
RESULTS_DIR=/home/hluo/scratch/results/test9
FOLDER_TYPE="test"
python ~/gapjncsegmentation/test.py --model_path $MODEL_PATH --data_dir $DATA_DIR --results_dir $RESULTS_DIR --folder_type $FOLDER_TYPE