#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --job-name=getpreds1072d
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=10:0:0
X_DIR=/home/hluo/scratch/select_dauer_data_512/original/train
Y_DIR=/home/hluo/scratch/select_dauer_data_512/ground_truth/train


X_DIR=/home/hluo/scratch/filtered_0_50_3x512x512/original/train
Y_DIR=/home/hluo/scratch/filtered_0_50_3x512x512/ground_truth/train



# Load model!!!
MODEL_NAME=model_job105
EPOCH=85
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth

# 0-50
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
X_DIR=/home/hluo/scratch/filtered_0_50_3x512x512/original
Y_DIR=/home/hluo/scratch/filtered_0_50_3x512x512/ground_truth
SAVECOMB=false
BATCH_SIZE=1
NUM_WORKERS=1
SAVE_VIS=false
SAVE2D=true
SAVE3D=false
SAVECOMB=false
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
USEALLSUBFOLDERS=true
SAVE_DIR=/home/hluo/scratch/0_50_${MODEL_NAME}
python /home/hluo/gapjncsegmentation/getpreds.py --useallsubfolders $USEALLSUBFOLDERS --x_dir $X_DIR --y_dir $Y_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --save_vis $SAVE_VIS --save2d $SAVE2D --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --savecomb $SAVECOMB


# 100-110
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
X_DIR=/home/hluo/scratch/100_110_3x512x512/original
Y_DIR=/home/hluo/scratch/100_110_3x512x512/ground_truth
BATCH_SIZE=1
NUM_WORKERS=4
SAVE_VIS=false
SAVE2D=true
SAVE3D=false
SAVECOMB=true
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
USEALLSUBFOLDERS=false
SAVE_DIR=/home/hluo/scratch/100_110_${MODEL_NAME}
PRED_MEMB=true
python /home/hluo/gapjncsegmentation/getpreds.py --pred_memb $PRED_MEMB --useallsubfolders $USEALLSUBFOLDERS --x_dir $X_DIR --y_dir $Y_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --save_vis $SAVE_VIS --save2d $SAVE2D --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --savecomb $SAVECOMB


# 111-120
X_DIR=/home/hluo/scratch/111_120_3x512x512_no_overlap/original
Y_DIR=/home/hluo/scratch/111_120_3x512x512_no_overlap/ground_truth
BATCH_SIZE=1
NUM_WORKERS=4
SAVE_VIS=false
SAVE2D=true
SAVE3D=false
SAVECOMB=true
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
USEALLSUBFOLDERS=false
SAVE_DIR=/home/hluo/scratch/111_120_${MODEL_NAME}
PRED_MEMB=true
python /home/hluo/gapjncsegmentation/getpreds.py --pred_memb $PRED_MEMB --useallsubfolders $USEALLSUBFOLDERS --x_dir $X_DIR --y_dir $Y_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --save_vis $SAVE_VIS --save2d $SAVE2D --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --savecomb $SAVECOMB

