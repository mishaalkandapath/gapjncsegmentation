import os
from torch.utils.data import DataLoader
from dataset import SliceDatasetWithFilename
import argparse
import time
from models import *
from utilities import *
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
    parser.add_argument('--save_vis', type=lambda x: (str(x).lower() == 'true'), default=True, help='save vis')
    parser.add_argument('--save2d', type=lambda x: (str(x).lower() == 'true'), default=True, help='save 2d')
    parser.add_argument('--savecomb', type=lambda x: (str(x).lower() == 'true'), default=True, help='save combined')
    parser.add_argument('--subvol_depth', type=int, default=3, help='num workers')
    parser.add_argument('--subvol_height', type=int, default=512, help='num workers')
    parser.add_argument('--subvol_width', type=int, default=512, help='num workers')
    args = parser.parse_args()
    
    print(f"Use2d {args.save2d}, savevis {args.save_vis}")

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
    batch_size = args.batch_size
    num_workers = args.num_workers

    print("Loading data dir")
    x_test_dir = args.x_dir
    y_test_dir = args.y_dir
    test_dataset = SliceDatasetWithFilename(x_test_dir, y_test_dir)
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

    total_precision = 0
    total_recall = 0
    start_time = time.time()
    subvol_depth, subvol_height, subvol_width = args.subvol_depth, args.subvol_height, args.subvol_width
    for i, data in enumerate(test_loader):
        inputs, labels, filenames = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        if i == 0:
            _, _, depth, height, width = inputs.shape # batch size, channels, depth, height, width

        # pad as needed
        print(inputs.shape)
        sub_vol_depth, sub_vol_height, sub_vol_width = inputs.shape[2:]
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
        binary_pred = torch.argmax(pred, dim=1) 

        precision = get_precision(pred=binary_pred, target=labels)
        recall = get_recall(pred=binary_pred, target=labels)
        print(f"precision {precision}, recall {recall}")
        # total_precision += precision
        # total_recall += recall
        
        # Save predictions for each epoch
        pred=pred[0, 1].detach().cpu()
        binary_pred=binary_pred[0].detach().cpu()
        labels = labels[0,0].detach().cpu()
        if args.savecomb:
            combined_volume = np.asarray((labels * 2 + binary_pred))
            vals, counts = np.unique(combined_volume, return_counts=True)
            color_combined_volume = get_colored_image(combined_volume)
            res = dict(map(lambda i,j : (i,j) , vals,counts))
            tn=counts[0]
            fp=counts[1]
            fn=counts[2]
            tp=counts[3]
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            total_precision += precision
            total_recall += recall
            print(f"comb precision {precision}, recall {recall}")
            
            print(vals, counts)
        binary_pred[binary_pred!=0]=255 # to visualize

        if args.save2d:
            for k in range(subvol_depth):
                cv2.imwrite(os.path.join(save_dir, "probpreds", f"{filenames[0]}_{k}.png"), np.array(pred[k]))
                cv2.imwrite(os.path.join(save_dir, "binarypreds", f"{filenames[0]}_{k}.png"), np.array(binary_pred[k]))
                if args.savecomb:
                    cv2.imwrite(os.path.join(save_dir, "combinedpreds", f"{filenames[0]}_{k}.png"), np.array(color_combined_volume[k]))
                    
        else:
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
                
        # if args.savecomb:
            
        avg_precision = total_precision / (i+1)
        avg_recall = total_recall/ (i+1)
        print(f"loaded {i} imgs: avg precision {avg_precision:.2f}, avg recall {avg_recall:.2f}")
        print(f"Time: {time.time()-start_time:.3f}s")

if __name__ == '__main__':
    main()