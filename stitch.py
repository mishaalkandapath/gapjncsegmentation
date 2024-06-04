import copy
import tqdm
import os
import numpy as np
import torchio as tio
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
                    
                small_3d_pred = np.array(small_3d_pred)
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