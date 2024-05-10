#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
#ADD OPTIONS HERE!!!!!!!
python /home/hluo/projects/def-mzhen/hluo/gapjncsegmentation/unet_3d.py --epochs 20 --batch_size 8 --lr 0.001 --data /home/hluo/projects/def-mzhen/hluo/gapjncsegmentation/data/small_data_256

