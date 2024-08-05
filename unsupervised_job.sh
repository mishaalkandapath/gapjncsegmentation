#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --job-name=bash
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=128G
#SBATCH --time=15:0:0
#SBATCH --signal=SIGUSR1@90
#SBATCH --account=def-mzhen
module purge
module load scipy-stack gcc cuda opencv
source ~/py10/bin/activate

wandb offline

#ADD OPTIONS HERE!!!!!!!
python contrastive.py --lightning --n_samples 32 --n_classes 36 --grad_cache