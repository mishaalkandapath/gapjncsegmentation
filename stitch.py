import copy
import tqdm
import os
import numpy as np
import torchio as tio
import cv2
import matplotlib.pyplot as plt
import argparse

def get_colored_image(image, color_map=None):
    # Define the default color map
    if color_map is None:
        color_map = {
            0: [0, 0, 0],  # Black (TN)
            1: [1, 0, 0],  # Red (FP) (pred only)
            2: [0, 0, 1],  # Blue (FN) (double_mask only)
            3: [0, 1, 0],  # Green (TP)
        }
    # Create an empty RGB image
    depth, height, width = image.shape[0], image.shape[1], image.shape[2]
    colored_image = np.zeros((depth, height, width, 3), dtype=np.float32) 
    
    # Map the pixel values to the corresponding colors
    for value, color in color_map.items():
        colored_image[image == value] = color
    return colored_image

def assemble_predictions(images_dir, preds_dir, gt_dir, start_s=0, start_y=0, start_x=0, end_s=6, end_y=8192, end_x=9216):
    tile_depth=3
    tile_width=512
    tile_height=512
    total_slices = ((end_s//tile_depth) * ((end_y-start_y)//tile_height )* ((end_x-start_x)//tile_width))
    slice_num = 0
    print(total_slices, "total slices")
    for s in range(start_s, end_s, 3):
        s_acc_img, s_acc_pred, s_acc_gt = [], [], []
        for y in range(start_y, end_y, 512):
            y_acc_img, y_acc_pred, y_acc_gt = [], [], []
            for x in range(start_x, end_x, 512):
                print(f"Processing volume {s,y,x} | Progress:{slice_num+1}/{total_slices} {(slice_num)/total_slices}", end="\r")
                suffix = r"z{}_y{}_x{}".format(s, y, x)
                
                # load img
                try:
                    img_vol = np.load(os.path.join(images_dir, f"{suffix}.npy"))
                except:
                    img_vol = np.zeros((3, 512,512))
                    print("no img")
                d, h, w = img_vol.shape
                if (d < tile_depth) or (h < tile_height) or (w < tile_width):
                    print("cropping since imgvol shape:", img_vol.shape)
                    img_vol = tio.CropOrPad((tile_depth, tile_height, tile_width))(torch.tensor(img_vol).unsqueeze(0))
                
                # load gt
                try:
                    gt_vol = np.load(os.path.join(gt_dir, f"{suffix}.npy"))
                except:
                    gt_vol = np.zeros((3,512,512))
                    print("no gt")
                if (d < tile_depth) or (h < tile_height) or (w< tile_width):
                    gt_vol = tio.CropOrPad((tile_depth, tile_height, tile_width))(gt_vol)
                
                # load pred
                try:
                    pred_vol = np.load(os.path.join(preds_dir, f"{suffix}.npy"))
                    pred_vol = np.argmax(pred_vol[0], 0) 
                except:
                    print("no pred vol")
                    pred_vol = np.zeros((3, 512,512))
                if (d < tile_depth) or (h < tile_height) or (w < tile_width):
                    pred_vol = tio.CropOrPad((tile_depth, tile_height, tile_width))(pred_vol)
                    
                small_3d_img = []
                small_3d_pred = []
                small_3d_gt = []
                for k in range(3):
                    img = img_vol[k]
                    gt = gt_vol[k]
                    pred = pred_vol[k]
                    small_3d_img += [img]
                    small_3d_gt += [gt]
                    small_3d_pred += [pred]
                    
                small_3d_pred = np.array(small_3d_pred) # (tile depth, tile height, tile width)
                small_3d_gt = np.array(small_3d_gt)
                small_3d_img = np.array(small_3d_img)
                    
                y_acc_gt += [small_3d_gt]
                y_acc_img += [small_3d_img]
                y_acc_pred += [small_3d_pred]
                slice_num+=1
            print(f"Processing volume {s,y,x} | Progress:{slice_num+1}/{total_slices} {(slice_num)/total_slices}")
            s_acc_img += [np.concatenate(y_acc_img, axis=2)]
            s_acc_pred += [np.concatenate(y_acc_pred, axis=2)]
            s_acc_gt += [np.concatenate(y_acc_gt, axis=2)]

        new_img = np.concatenate(s_acc_img, axis=1)
        new_pred = np.concatenate(s_acc_pred, axis=1)
        new_gt = np.concatenate(s_acc_gt, axis=1)
        
        return new_img, new_pred, new_gt
    
    
parser = argparse.ArgumentParser(description="Get evaluation metrics for the model")
parser.add_argument("--save_dir", type=str, required=True, help="Path to the model file")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--use_lines", type=bool, default=False, help="Path to the results directory")
parser.add_argument("--show_img", type=bool, default=False, help="Whether this is train, test, or valid folder")
parser.add_argument("--alpha", type=float, default=0.4, help="Whether this is train, test, or valid folder")
parser.add_argument("--line_width", type=int, default=3, help="Whether this is train, test, or valid folder")
args = parser.parse_args()
# constants
data_dir= args.data_dir
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
img_dir=os.path.join(data_dir, "original")
gt_dir=os.path.join(data_dir, "ground_truth")
pred_dir=os.path.join(data_dir, "pred")
use_lines = args.use_lines
show_img = args.show_img
alpha = args.alpha
line_width = args.line_width
yellow = [255, 255, 0]
print(data_dir)
print(save_dir)
print("showimg", show_img)
print("uselines", use_lines)


# stich together preds
new_img, new_pred, new_gt = assemble_predictions(img_dir, pred_dir, gt_dir)
binary_gt = new_gt.copy()
binary_gt[binary_gt != 0] = 1
volume_depth = new_img.shape[0]
print("shape", new_img.shape, new_pred.shape, new_gt.shape)
unique_gt_labels = np.unique(new_gt)
    
# save by confidence
height, width = new_img.shape[1], new_img.shape[2]
for label in unique_gt_labels:
    print("Saving confidence ", label)
    confidence_gt = new_gt.copy()
    confidence_gt[confidence_gt != label] = 0
    combined_volume = np.asarray((confidence_gt[1] * 2 + new_pred))
    color_combined_volume = get_colored_image(combined_volume)
    if show_img:
        for k in range(volume_depth):
            print(f"Saved {k}")
            fig = plt.figure(num=1)
            if use_lines:
                lined_img = new_img[k].copy()
                lined_img = (lined_img - np.min(lined_img))/(np.max(lined_img)-np.min(lined_img))
                lined_img = np.stack((lined_img,) * 3, axis=-1)
                for y in range(0, height, 512):
                    lined_img[y:y+line_width, :] = yellow
                for x in range(0, width, 512):
                    lined_img[:, x:x+line_width] = yellow
                plt.imshow(lined_img)
            else:
                plt.imshow(new_img[k], cmap='gray')
            plt.imshow(color_combined_volume[k], alpha=alpha)
            plt.savefig(save_dir + f"combwithimg_confidence{str(int(label))}_slice{k}.png", dpi=800)
            plt.close("all")
    else:
        for k in range(volume_depth):
            plt.imsave(save_dir + f"comb_confidence{str(int(label))}_slice{k}.png", color_combined_volume[k], cmap="gray")
            print(f"Saved slice {k}")

    
# save with all labels
combined_volume = np.asarray((binary_gt[1] * 2 + new_pred))
color_combined_volume = get_colored_image(combined_volume)
if show_img:
    for k in range(volume_depth):
        print(f"Saved {k}")
        fig = plt.figure(num=1)
        if use_lines:
            lined_img = new_img[k].copy()
            lined_img = (lined_img - np.min(lined_img))/(np.max(lined_img)-np.min(lined_img))
            lined_img = np.stack((lined_img,) * 3, axis=-1)
            for y in range(0, height, 512):
                lined_img[y:y+line_width, :] = yellow
            for x in range(0, width, 512):
                lined_img[:, x:x+line_width] = yellow
            plt.imshow(lined_img)
        else:
            plt.imshow(new_img[k], cmap='gray')
        plt.imshow(color_combined_volume[k], alpha=alpha)
        plt.savefig(save_dir + f"combwithimg_slice{k}.png", dpi=800)
        plt.close("all")
else:
    for k in range(volume_depth):
        plt.imsave(save_dir + f"comb_slice{k}.png", color_combined_volume[k], cmap="gray")
        print(f"Saved slice {k}")
    