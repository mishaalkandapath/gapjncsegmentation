#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=128G
#SBATCH --time=20:0:0
#SBATCH --signal=SIGUSR1@90
#SBATCH --account=def-mzhen
module purge
source ~/py10/bin/activate
module load scipy-stack gcc cuda opencv

#ADD OPTIONS HERE!!!!!!!
#python unet.py --dataset new --mask_neurons #--mask_mito
#python unet.py --dataset new --gendice --mask_neurons
#python unet.py --dataset new --mask_neurons --td --batch_size 5
#python unet.py --split --dataset tiny --mask_neurons --mask_mito --batch_size 32
#python /home/mishaalk/projects/def-mzhen/mishaalk/gapjncsegmentation/unet.py --split --dataset tiny --batch_size 100

python unet.py --dataset new3d --td --gendice --batch_size 10
#python unet.py --dataset new3d --td --batch_size 10 -- 24 hours!
#python unet.py --dataset tiny --dice --batch_size 100 --mask_neurons -- not yet!
#python unet.py --dataset new --mask_neurons --gendice  10 relly but 12 hrs?