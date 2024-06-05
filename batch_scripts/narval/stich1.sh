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
DATA_DIR="/home/hluo/scratch/results/savepred6/"
SAVE_DIR="/home/hluo/scratch/stichsavepred6/"
USE_LINES=true
SHOW_IMG=true
python ~/gapjncsegmentation/stitch.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES