import torchio as tio
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
import tqdm
from models import *
from utilities import *


mask_dir="/Volumes/LaCie/sem_dauer_2_gj_gt"
img_dir="/Volumes/LaCie/SEM_dauer_2_em"
model_name = "model_job84"
epoch=49
fp=f"/Volumes/LaCie/models/{model_name}_epoch_{epoch}.pth"

# -- get imgs & mask fp
img_files = os.listdir(img_dir)
mask_files = os.listdir(mask_dir)
mask_pattern=r"sem_dauer_2_gj_gt_s(\d+).png"
img_pattern=r"SEM_dauer_2_em_s(\d+).png"
img_files = [os.path.join(img_dir, f) for f in img_files if f.endswith(".png")]
mask_files = [os.path.join(mask_dir, f) for f in mask_files if f.endswith(".png")]

# -- load model
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model, optimizer, epoch, loss, batch_size, lr, focal_loss_weights = load_checkpoint(model, optimizer, fp)
model = model.eval()
print("model loaded")

# -- get masks:
z_indices = [101, 102, 103, 104, 105, 106, 107, 108, 109]
width = 512
height = 512
depth = len(z_indices)
new_height = 256 # new height of each tile
new_width = 256 # new height of each tile

start_z = 0
start_y = 0
start_x = 0
ending_depth = full_volume_img.shape[0]
ending_height = full_volume_img.shape[1]
ending_width = full_volume_img.shape[2]
print(ending_depth, ending_height, ending_width)
subvol_depth = 3
subvol_height = 256
subvol_width = 256

while start_z < ending_depth:
    end_z = start_z + subvol_depth
    while start_y < ending_height:
        end_y = start_y + subvol_height 
        while start_x < ending_width:
            end_x = start_x + subvol_width
            sub_volume_img = full_volume_img[start_z:end_z, start_y:end_y, start_x:end_x]
            sub_volume_mask = full_volume_mask[start_z:end_z, start_y:end_y, start_x:end_x]
            sub_vol_depth, sub_vol_height, sub_vol_width = sub_volume_img.shape
            image = torch.tensor(sub_volume_img).float().unsqueeze(0)
            if (sub_vol_height < subvol_height) or (sub_vol_width < subvol_width) or (sub_vol_depth < subvol_depth):
                image = tio.CropOrPad((subvol_depth, subvol_height, subvol_width))(image)
            image = tio.ZNormalization()(image)
            print(image.shape)
            intermed_pred, sub_volume_pred = model(image)
            binary_pred = torch.argmax(sub_volume_pred[0], dim=0) # (depth, height, width)
            np.save(os.path.join(save_dir, "original", f"z{start_z}_y{start_y}_x{start_x}.npy"), sub_volume_img)
            np.save(os.path.join(save_dir, "ground_truth", f"z{start_z}_y{start_y}_x{start_x}.npy"), sub_volume_mask)
            np.save(os.path.join(save_dir, "pred", f"z{start_z}_y{start_y}_x{start_x}.npy"), sub_volume_pred.detach().cpu())            
            fig, ax = plt.subplots(3, subvol_depth, figsize=(15,5), num=1)
            visualize_3d_slice(sub_volume_img, ax[0], "Image")
            visualize_3d_slice(sub_volume_mask, ax[1], "Mask")
            visualize_3d_slice(binary_pred, ax[2], "Pred")
            print(f"Saved z{start_z}-{end_z} y{start_y}-{end_y} x{start_x}-{end_x} subvolume")
            plt.savefig(os.path.join(save_dir, "visualize", f"z{start_z}_y{start_y}_x{start_x}.png"))
            plt.close("all")
            start_x = end_x
        start_y = end_y
    start_z = end_z