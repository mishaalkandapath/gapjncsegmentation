#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=128G
#SBATCH --time=10:0:0
#SBATCH --signal=SIGUSR1@90
#SBATCH --account=def-mzhen
module purge
source ~/py10/bin/activate
module load scipy-stack gcc cuda opencv

#ADD OPTIONS HERE!!!!!!!
#python unet.py --dataset new --mask_neurons #--mask_mito
python unet.py --split --dataset tiny --mask_neurons --mask_mito --batch_size 32
#python /home/mishaalk/projects/def-mzhen/mishaalk/gapjncsegmentation/unet.py --split --dataset tiny --batch_size 100
