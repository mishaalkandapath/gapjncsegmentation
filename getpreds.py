""" 
getpreds.py
Given a directory of subvolumes, this script will load the model and predict on each subvolume, and generate test metrics such as precision and recall.

Sample usage:
MODEL_NAME=model_job203b
EPOCH=310
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
X_DIR=/home/hluo/scratch/data/111_120_3x512x512/original
Y_DIR=/home/hluo/scratch/data/111_120_3x512x512/ground_truth
SAVE_DIR=/home/hluo/scratch/preds/111_120_${MODEL_NAME}_epoch_${EPOCH}
SAVE2D=true
SAVECOMB=false
USEALLSUBFOLDERS=false
PRED_MEMB=false
BATCH_SIZE=1
NUM_WORKERS=4
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
DOWNSAMPLE_FACTOR=2
UPSAMPLE=true
python /home/hluo/gapjncsegmentation/getpreds.py --upsample $UPSAMPLE --downsample_factor $DOWNSAMPLE_FACTOR --pred_memb $PRED_MEMB --useallsubfolders $USEALLSUBFOLDERS --x_dir $X_DIR --y_dir $Y_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --save2d $SAVE2D --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --savecomb $SAVECOMB


(no downsample)
module purge
source ~/py39/bin/activate
module load scipy-stack gcc cuda opencv
MODEL_NAME=model_job204c
EPOCH=52
SLICES="100_110"
MODEL_PATH=/home/hluo/scratch/models/${MODEL_NAME}/${MODEL_NAME}_epoch_${EPOCH}.pth
X_DIR=/home/hluo/scratch/data/${SLICES}_3x512x512/original
Y_DIR=/home/hluo/scratch/data/${SLICES}_3x512x512/ground_truth
SAVE_DIR=/home/hluo/scratch/preds/${SLICES}_${MODEL_NAME}_epoch_${EPOCH}
SAVE2D=true
SAVECOMB=false
USEALLSUBFOLDERS=false
PRED_MEMB=false
BATCH_SIZE=1
NUM_WORKERS=4
SUBVOL_DEPTH=3
SUBVOL_HEIGHT=512
SUBVOL_WIDTH=512
python /home/hluo/gapjncsegmentation/getpreds.py --pred_memb $PRED_MEMB --useallsubfolders $USEALLSUBFOLDERS --x_dir $X_DIR --y_dir $Y_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --save2d $SAVE2D --subvol_depth $SUBVOL_DEPTH --subvol_height $SUBVOL_HEIGHT --subvol_width $SUBVOL_WIDTH --savecomb $SAVECOMB

"""

import os
from torch.utils.data import DataLoader
from utilities.dataset import SliceDatasetWithFilename, SliceDatasetWithFilenameAllSubfolders
import argparse
import time
from utilities.models import *
from utilities.utilities import *
import torchio as tio
from scipy.ndimage import label



def get_entity_metrics(pred_vol, gt_vol, entity_thresh=0.1):
    """ returns fp, fn, tp """
    labeled_array, num_features = label(pred_vol)
    print(f"Number of entities found in pred: {num_features}")
    tp = 0
    fp = 0
    fn = 0
    for i in range(1, num_features + 1):
        component = (labeled_array == i).astype(np.uint8) * 255
        num_intersect = np.count_nonzero(component[gt_vol != 0]) # nonzero pixels in component where gt is also nonzero
        num_pred = np.count_nonzero(component)
        proportion_intersect = num_intersect/num_pred
        if proportion_intersect >= entity_thresh:
            tp += 1
        else:
            fp += 1

    labeled_array, num_gt_features = label(gt_vol)
    print(f"Number of entities found in gt: {num_gt_features}")
    tp_intersect_gt = 0
    for i in range(1, num_gt_features + 1):
        component = (labeled_array == i).astype(np.uint8) * 255
        num_intersect = np.count_nonzero(component[pred_vol != 0])
        num_gt = np.count_nonzero(component)
        if num_intersect/num_gt >= entity_thresh:
            tp_intersect_gt += 1
        else:
            fn += 1
    return fp, fn, tp

def main():
    print("starting...")
    parser = argparse.ArgumentParser(description="save preds")
    parser.add_argument('--x_dir', type=str, required=True, help='data dir for testing data')
    parser.add_argument('--y_dir', type=str, required=True, help='data dir for testing data')
    parser.add_argument('--save_dir', type=str, required=True, help='save dir for testing data')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--threshold', type=int, default=0.5, help='threshold for binary pred')
    parser.add_argument('--pred_memb', type=lambda x: (str(x).lower() == 'true'), default=False, help='if model also predicts membrane')
    parser.add_argument('--save2d', type=lambda x: (str(x).lower() == 'true'), default=True, help='save 2d')
    parser.add_argument('--savecomb', type=lambda x: (str(x).lower() == 'true'), default=False, help='save combined')
    parser.add_argument('--useallsubfolders', type=lambda x: (str(x).lower() == 'true'), default=False, help='use all subfolders')
    parser.add_argument('--upsample', type=lambda x: (str(x).lower() == 'true'), default=False, help='use all subfolders')
    parser.add_argument('--use_entity_metrics', type=lambda x: (str(x).lower() == 'true'), default=False, help='use all subfolders')
    parser.add_argument('--subvol_depth', type=int, default=3, help='num workers')
    parser.add_argument('--subvol_height', type=int, default=512, help='num workers')
    parser.add_argument('--subvol_width', type=int, default=512, help='num workers')
    parser.add_argument('--downsample_factor', type=int, default=1, help='num workers')
    args = parser.parse_args()
    
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # make save dir
    save_dir = args.save_dir
    save_dir_binary = f"{save_dir}_binary"
    save_dir_comb = f"{save_dir}_comb"
    if args.save2d and not os.path.exists(save_dir_binary):
        os.makedirs(save_dir_binary)
    if args.savecomb and not os.path.exists(save_dir_comb):
        os.makedirs(save_dir_comb)
        

    print("----------------------------Loading data dir----------------------------")
    batch_size = args.batch_size
    num_workers = args.num_workers
    downsample_factor = args.downsample_factor
    x_test_dir = args.x_dir
    y_test_dir = args.y_dir
    if args.useallsubfolders:
        print("using all subfolders")
        test_dataset = SliceDatasetWithFilenameAllSubfolders(x_test_dir, y_test_dir, downsample_factor=downsample_factor)
    else:
        if args.upsample:
            test_dataset = SliceDatasetWithFilename(x_test_dir, y_test_dir, downsample_factor=downsample_factor, downsample_mask=False)
        else:
            test_dataset = SliceDatasetWithFilename(x_test_dir, y_test_dir, downsample_factor=downsample_factor, downsample_mask=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) # change num_workers as needed
    print(f"Batch size: {batch_size}, Number of workers: {num_workers}")
    total_imgs = len(test_dataset)
    print(f"Data loaders created. Train dataset size: {total_imgs}")


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    model = UNet()
    model = model.to(DEVICE)
    load_model_path = args.model_path
    if load_model_path is not None:
        checkpoint = torch.load(load_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_model_path}.")
    print(f"Model is on device {next(model.parameters()).device}")
    model = model.eval()


    print("----------------------------Generating predictions----------------------------")
    start_time = time.time()
    total_tp=0
    total_fp=0
    total_fn=0
    total_tn=0
    for i, data in enumerate(test_loader):
        inputs, labels, filenames = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        print(inputs.shape, labels.shape)
            
        # pad image
        subvol_depth, subvol_height, subvol_width = args.subvol_depth, args.subvol_height, args.subvol_width
        if downsample_factor > 1:
            subvol_height, subvol_width = subvol_height // downsample_factor, subvol_width // downsample_factor
        d, h, w = inputs.shape[2:]
        if (h < subvol_height) or (w < subvol_width) or (d < subvol_depth):
            tmp = tio.CropOrPad((subvol_depth, subvol_height, subvol_width))(inputs[0].detach().cpu())
            inputs = tmp.unsqueeze(0)
            inputs = inputs.to(DEVICE)
            print("Padded to", inputs.shape)
            del tmp

        # pad labels if necessary
        d, h, w = labels.shape[2:]
        if args.upsample:
            subvol_depth, subvol_height, subvol_width = args.subvol_depth, args.subvol_height, args.subvol_width # keep mask original shape
        if (h < subvol_height) or (w < subvol_width) or (d < subvol_depth):
            tmp = tio.CropOrPad((subvol_depth, subvol_height, subvol_width))(labels[0].detach().cpu())
            labels = tmp.unsqueeze(0)
            labels = labels.to(DEVICE)
            print("Padded to", labels.shape)
            del tmp
            
        # inputs: (batch_size=1, channels=1, depth, height, width)
        interm_pred, pred = model(inputs)
        print("pred", pred.shape)
        
        # take argmax
        if args.pred_memb:
            threshold=args.threshold
            binary_pred = pred[0, 1].detach().cpu()
            binary_pred[binary_pred >= threshold] = 1
            binary_pred[binary_pred < threshold] = 0
        else:
            binary_pred = torch.argmax(pred, dim=1) 
            pred=pred[0, 1].detach().cpu()
            binary_pred=binary_pred[0].detach().cpu()
            binary_pred[binary_pred != 0] = 1
            print("binary_pred", binary_pred.shape)
            
            # binary_pred (depth, height, width)
        # downsample so upsample
        if args.upsample:
            print("before", binary_pred.shape)
            # convert to float, not long
            binary_pred = binary_pred.float().unsqueeze(0) # (batch_size, channels, depth, height, width) -> (channels, depth, height, width)
            print(binary_pred.shape)
            binary_pred = nn.Upsample(scale_factor=downsample_factor, mode='nearest')(binary_pred) # (upsample takes 4D input)
            binary_pred = binary_pred[0]
            print("upsampled", binary_pred.shape) # (depth, height, width)
            
            
        if args.use_entity_metrics:
            entity_fp, entity_fn, entity_tp = get_entity_metrics(binary_pred, labels, args.entity_thresh)
            
        
        labels = labels[0,0].detach().cpu()
        combined_volume = np.asarray((labels * 2 + binary_pred))
        vals, counts = np.unique(combined_volume, return_counts=True)
        res = dict(map(lambda i,j : (int(i),j) , vals,counts))
        fp = res.get(1, 0)
        fn = res.get(2, 0)
        tp = res.get(3, 0)
        tn = res.get(0, 0)
        total_tp+=tp
        total_fp+=fp
        total_fn+=fn
        total_tn+=tn
        precision=tp/(tp+fp) if (tp + fp) != 0 else 999
        recall=tp/(tp+fn) if (tp + fn) != 0 else 999
        print(vals, counts)
        print(f"comb precision {precision}, recall {recall}")
        
        if args.savecomb:
            color_combined_volume = get_colored_image(combined_volume)
        binary_pred[binary_pred!=0]=255 # to visualize
        if args.save2d:
            for k in range(subvol_depth):
                cv2.imwrite(os.path.join(save_dir_binary, f"{filenames[0]}_{k}.png"), np.array(binary_pred[k]))
                if args.savecomb:
                    cv2.imwrite(os.path.join(save_dir_comb, f"{filenames[0]}_{k}.png"), np.array(color_combined_volume[k]))
            
        avg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) !=0 else 999
        avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) !=0 else 999
        print(f"---------------------- (tn) {total_tn}(fp) {total_fp} (fn) {total_fn} (tp) {total_tp}")
        print(f"----------------------loaded {i}/{total_imgs} imgs: avg precision {avg_precision:.3f}, avg recall {avg_recall:.3f}----------------------")
        print(f"Time: {time.time()-start_time:.3f}s")

if __name__ == '__main__':
    main()