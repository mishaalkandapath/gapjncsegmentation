#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=unet
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=128G
#SBATCH --time=6:30:0
#SBATCH --signal=SIGUSR1@90
#SBATCH --account=def-mzhen
module purge
module load scipy-stack gcc cuda opencv
source ~/py10/bin/activate

wandb offline

#ADD OPTIONS HERE!!!!!!!
#python unet.py --dataset new --mask_neurons #--mask_mito
#python unet.py --dataset new --gendice --mask_neurons
#python unet.py --dataset new --mask_neurons --td --batch_size 5
#python unet.py --split --dataset tiny --mask_neurons --mask_mito --batch_size 32
#python /home/mishaalk/projects/def-mzhen/mishaalk/gapjncsegmentation/unet.py --split --dataset tiny --batch_size 100

# python unet.py --dataset new3d --td --gendice --batch_size 10
python unet.py --dataset new3d --td --batch_size 10 --dicefocal --focalweight 0.2 #-- 14 hours!
#python unet.py --dataset tiny --dice --batch_size 100 --mask_neurons -- 4 hrs
#python unet.py --dataset new --mask_neurons --gendice 10 hrs?
# python unet.py --dataset tiny --batch_size 100 --mask_neurons --mask_mito --epochs 500 EPOCHS REMEMVER TO CHANGE - about 10 hrs
#