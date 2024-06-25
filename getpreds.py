""" 
getpreds.py
Given a directory of subvolumes, this script will load the model and predict on each subvolume, and generate test metrics such as precision and recall.

Sample usage:
X_DIR="data/tiniest_data_64"
Y_DIR="data/tiniest_data_64"
SAVE_DIR="data/tiniest_data_64"
MODEL_PATH="model_job84"
python getpreds.py --x_dir $X_DIR --y_dir $Y_DIR --save_dir $SAVE_DIR --model_path $MODEL_PATH
"""

import os
from torch.utils.data import DataLoader
from utilities.dataset import SliceDatasetWithFilename, SliceDatasetWithFilenameAllSubfolders
import argparse
import time
from utilities.models import *
from utilities.utilities import *
import torchio as tio

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
    parser.add_argument('--save_vis', type=lambda x: (str(x).lower() == 'true'), default=True, help='save vis')
    parser.add_argument('--save2d', type=lambda x: (str(x).lower() == 'true'), default=True, help='save 2d')
    parser.add_argument('--save3d', type=lambda x: (str(x).lower() == 'true'), default=False, help='save 3d')
    parser.add_argument('--savecomb', type=lambda x: (str(x).lower() == 'true'), default=False, help='save combined')
    parser.add_argument('--useallsubfolders', type=lambda x: (str(x).lower() == 'true'), default=False, help='use all subfolders')
    parser.add_argument('--subvol_depth', type=int, default=3, help='num workers')
    parser.add_argument('--subvol_height', type=int, default=512, help='num workers')
    parser.add_argument('--subvol_width', type=int, default=512, help='num workers')
    parser.add_argument('--downsample_factor', type=int, default=1, help='num workers')
    args = parser.parse_args()
    
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # make save dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "binarypreds")):
        os.makedirs(os.path.join(save_dir, "binarypreds"))
    if not os.path.exists(os.path.join(save_dir, "probpreds")):
        os.makedirs(os.path.join(save_dir, "probpreds"))
    if not os.path.exists(os.path.join(save_dir, "visualize")):
        os.makedirs(os.path.join(save_dir, "visualize"))
    if not os.path.exists(os.path.join(save_dir, "combinedpreds")):
        os.makedirs(os.path.join(save_dir, "combinedpreds"))
    

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
        test_dataset = SliceDatasetWithFilename(x_test_dir, y_test_dir, downsample_factor=downsample_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) # change num_workers as needed
    print(f"Batch size: {batch_size}, Number of workers: {num_workers}")
    print(f"Data loaders created. Train dataset size: {len(test_dataset)}")


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
    subvol_depth, subvol_height, subvol_width = args.subvol_depth, args.subvol_height, args.subvol_width
    total_tp=0
    total_fp=0
    total_fn=0
    total_tn=0
    for i, data in enumerate(test_loader):
        inputs, labels, filenames = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        if i == 0:
            _, _, depth, height, width = inputs.shape # batch size, channels, depth, height, width
        print(inputs.shape)
        sub_vol_depth, sub_vol_height, sub_vol_width = inputs.shape[2:]
        
        # pad image and label
        if (sub_vol_height < subvol_height) or (sub_vol_width < subvol_width) or (sub_vol_depth < subvol_depth):
            tmp = tio.CropOrPad((subvol_depth, subvol_height, subvol_width))(inputs[0].detach().cpu())
            inputs = tmp.unsqueeze(0)
            inputs = inputs.to(DEVICE)
            print("Padded to", inputs.shape)
            del tmp

        sub_vol_depth, sub_vol_height, sub_vol_width = labels.shape[2:]
        if (sub_vol_height < subvol_height) or (sub_vol_width < subvol_width) or (sub_vol_depth < subvol_depth):
            tmp = tio.CropOrPad((subvol_depth, subvol_height, subvol_width))(labels[0].detach().cpu())
            labels = tmp.unsqueeze(0)
            labels = labels.to(DEVICE)
            print("Padded to", labels.shape)
            del tmp
            
        interm_pred, pred = model(inputs)
        
        # take argmax
        if args.pred_memb:
            threshold=args.threshold
            binary_pred = pred[0, 1].detach().cpu()
            binary_pred[binary_pred >= threshold] = 1
            binary_pred[binary_pred < threshold] = 0
            pred = pred[0, 0].detach().cpu()
        else:
            binary_pred = torch.argmax(pred, dim=1) 
            pred=pred[0, 1].detach().cpu()
            binary_pred=binary_pred[0].detach().cpu()
            binary_pred[binary_pred != 0] = 1
            
        # downsample so upsample
        if downsample_factor > 1:
            
            
        
        labels = labels[0,0].detach().cpu()
        combined_volume = np.asarray((labels * 2 + binary_pred))
        vals, counts = np.unique(combined_volume, return_counts=True)
        res = dict(map(lambda i,j : (int(i),j) , vals,counts))
        fp = res.get(1, 0)
        fn = res.get(2, 0)
        tp = res.get(3, 0)
        total_tp+=tp
        total_fp+=fp
        total_fn+=fn
        precision=tp/(tp+fp) if (tp + fp) != 0 else 0
        recall=tp/(tp+fn) if (tp + fn) != 0 else 0
        print(vals, counts)
        print(f"comb precision {precision}, recall {recall}")
        print(vals, counts)
        if args.savecomb:
            color_combined_volume = get_colored_image(combined_volume)
        binary_pred[binary_pred!=0]=255 # to visualize

        if args.save2d:
            for k in range(subvol_depth):
                cv2.imwrite(os.path.join(save_dir, "probpreds", f"{filenames[0]}_{k}.png"), np.array(pred[k]))
                cv2.imwrite(os.path.join(save_dir, "binarypreds", f"{filenames[0]}_{k}.png"), np.array(binary_pred[k]))
                if args.savecomb:
                    cv2.imwrite(os.path.join(save_dir, "combinedpreds", f"{filenames[0]}_{k}.png"), np.array(color_combined_volume[k]))
                    
        elif args.save3d:
            print(f"Saving {filenames[0]} 3D")
            np.save(os.path.join(save_dir, "probpreds", f"{filenames[0]}.npy"), pred)
            np.save(os.path.join(save_dir, "binarypreds", f"{filenames[0]}.npy"), binary_pred)
            if args.save_vis:
                if i==0:
                    print("Saving visualizations", args.save_vis)
                fig, ax = plt.subplots(4, depth, figsize=(15, 8), num=1)
                for j in range(depth):
                    ax[0, j].imshow(inputs[0, 0, j].detach().cpu(), cmap="gray")
                    ax[1, j].imshow(labels[j], cmap="gray")
                    ax[2, j].imshow(binary_pred[j], cmap="gray")
                    im = ax[3, j].imshow(pred[j], cmap="viridis")
                cbar = fig.colorbar(im, ax=ax[3, :])
                ax[0, 0].set_ylabel("Input")
                ax[1, 0].set_ylabel("Ground Truth")
                ax[2, 0].set_ylabel("Prediction")
                ax[3, 0].set_ylabel("Prediction (probability)")
                plt.savefig(os.path.join(save_dir, "visualize", f"{filenames[0]}.png"))
        else:
            pass
                
            
        avg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) !=0 else 0
        avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) !=0 else 0
        print(f"----------------------loaded {i} imgs: avg precision {avg_precision:.3f}, avg recall {avg_recall:.3f}----------------------")
        print(f"Time: {time.time()-start_time:.3f}s")

if __name__ == '__main__':
    main()