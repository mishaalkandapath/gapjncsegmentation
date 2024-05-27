#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=job80
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --mem=16G
#SBATCH --time=11:0:0
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --gpus-per-node=1

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done

module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
EPOCHS=30
BATCH_SIZE=1
NUM_WORKERS=4
LR=0.001
DATA_DIR=/home/hluo/scratch/select_dauer_data_512
MODEL_DIR=/home/hluo/scratch/models
MODEL_NAME="model_job80"
LOAD_MODEL_PATH=/home/hluo/scratch/models/model_job20c/model_job20c_epoch_99.pth
ALPHA=0.96
GAMMA=2
NUM_PREDICTIONS_TO_LOG=10
python /home/hluo/gapjncsegmentation/unet_3d.py --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --data_dir $DATA_DIR --model_name $MODEL_NAME --num_workers $NUM_WORKERS --model_dir $MODEL_DIR --alpha $ALPHA --gamma $GAMMA --num_predictions_to_log $NUM_PREDICTIONS_TO_LOG --load_model_path $LOAD_MODEL_PATH
