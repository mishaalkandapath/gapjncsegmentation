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
DATA_DIR="/Volumes/LaCie/may31/savepred3/"
SAVE_DIR="/Volumes/LaCie/gapjnc93/"

DATA_DIR="/Volumes/LaCie/june4/savepred6/"
SAVE_DIR="/Volumes/LaCie/gapjncsave6/"
USE_LINES=false
SHOW_IMG=false
python stitch.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES


DATA_DIR=/home/hluo/scratch/filtered_100_110_3x512x512_40/original/train
f
DATA_DIR=/home/hluo/scratch/filtered_0_50_3x512x512/original/train


DATA_DIR=/home/hluo/scratch/filtered_100_110_3x512x512_40/original/valid

DATA_DIR=/home/hluo/scratch/filtered_0_50_3x512x512/original/valid
python ~/gapjncsegmentation/remove_invalid_samples.py --data_dir $DATA_DIR --depth 3 --height 512 --width 512

python ~/gapjncsegmentation/stitch.py --data_dir $DATA_DIR --save_dir $SAVE_DIR --show_img $SHOW_IMG --use_lines $USE_LINES