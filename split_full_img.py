import torchio as tio
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from models import *
from utilities import *

print("starting...")
parser = argparse.ArgumentParser(description="save preds")
parser.add_argument('--start_x', type=int, default=0, help='Starting X coordinate')
parser.add_argument('--start_y', type=int, default=0, help='Starting Y coordinate')
parser.add_argument('--start_z', type=int, default=100, help='Starting Z coordinate')
parser.add_argument('--ending_depth', type=int, default=111, help='Ending depth')
parser.add_argument('--ending_height', type=int, default=8328, help='Ending height')
parser.add_argument('--ending_width', type=int, default=9360, help='Ending width')
parser.add_argument('--subvol_depth', type=int, default=3, help='subvol depth')
parser.add_argument('--subvol_height', type=int, default=256, help='subvol height')
parser.add_argument('--subvol_width', type=int, default=256, help='subvol width')
parser.add_argument('--step_z', type=int, default=3, help='step depth')
parser.add_argument('--step_y', type=int, default=256, help='step height')
parser.add_argument('--step_x', type=int, default=256, help='step width')
parser.add_argument('--mask_dir', type=str, default="/Volumes/LaCie/sem_dauer_2_gj_gt", help='Mask directory')
parser.add_argument('--img_dir', type=str, default="/Volumes/LaCie/SEM_dauer_2_em", help='Image directory')
parser.add_argument('--save_dir', type=str, default="", help='Save directory')
parser.add_argument('--model_path', type=str, default="model_job84", help='Model name')
parser.add_argument('--use_full_volume', type=bool, default=True, help='use full volume')
args = parser.parse_args()

# -- args
start_x = args.start_x
start_y = args.start_y
start_z = args.start_z
ending_depth = args.ending_depth
ending_height = args.ending_height
ending_width = args.ending_width
subvol_depth = args.subvol_depth
subvol_height = args.subvol_height
subvol_width = args.subvol_width
mask_dir= args.mask_dir
img_dir= args.img_dir
fp=args.model_path
save_dir=args.save_dir
if not os.path.exists(os.path.join(save_dir, "original")): 
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "ground_truth"), exist_ok=True)
    
# -- get imgs & mask fp
img_files = os.listdir(img_dir)
mask_files = os.listdir(mask_dir)
mask_pattern=r"sem_dauer_2_gj_gt_s(\d+).png"
img_pattern=r"sem_dauer_2_em_s(\d+).png"
img_files = [os.path.join(img_dir, f) for f in img_files if f.endswith(".png")]
mask_files = [os.path.join(mask_dir, f) for f in mask_files if f.endswith(".png")]
print("# imgs: ", len(img_files), "# masks: ", len(mask_files))
# -- get masks:
i=0
for z in range(start_z, ending_depth):
    tmp_img = get_img_by_z(z, img_files, img_pattern)
    tmp_mask = get_img_by_z(z, mask_files, mask_pattern)
    if i == 0:
        h, w = tmp_img.shape[0], tmp_img.shape[1]
        full_volume_img = np.zeros((ending_depth-start_z, h, w))
        full_volume_mask = np.zeros((ending_depth-start_z, h, w))
        print("Created full img & mask", full_volume_img.shape, full_volume_mask.shape)
    full_volume_img[i] = tmp_img
    full_volume_mask[i] = tmp_mask
    i+=1
    print(f"done {z} z-slice")
print("full volume shape:", full_volume_img.shape, full_volume_mask.shape)

start_z = 0 # start from 0 of local vol
# ending_depth = full_volume_img.shape[0]
if args.use_full_volume:
    ending_height = full_volume_img.shape[1]
    ending_width = full_volume_img.shape[2]
print("subvolume shape:", subvol_depth, subvol_height, subvol_width)
print("ending:", ending_depth, ending_height, ending_width)

step_z = args.step_z
step_y = args.step_y
step_x = args.step_x
while start_z < ending_depth:
    end_z = start_z + subvol_depth
    start_y = args.start_y
    while start_y < ending_height:
        end_y = start_y + subvol_height 
        start_x = args.start_x
        while start_x < ending_width:
            end_x = start_x + subvol_width
            sub_volume_img = full_volume_img[start_z:end_z, start_y:end_y, start_x:end_x]
            sub_volume_mask = full_volume_mask[start_z:end_z, start_y:end_y, start_x:end_x] # with confidence levels
            sub_vol_depth, sub_vol_height, sub_vol_width = sub_volume_img.shape
            image = torch.tensor(sub_volume_img).float().unsqueeze(0)
            if (sub_vol_height < subvol_height) or (sub_vol_width < subvol_width) or (sub_vol_depth < subvol_depth):
                image = tio.CropOrPad((subvol_depth, subvol_height, subvol_width))(image)
                
            try:
                np.save(os.path.join(save_dir, "original", f"z{start_z}_y{start_y}_x{start_x}.npy"), sub_volume_img)
                np.save(os.path.join(save_dir, "ground_truth", f"z{start_z}_y{start_y}_x{start_x}.npy"), sub_volume_mask)
                print(f"Saved z{start_z}-{end_z} y{start_y}-{end_y} x{start_x}-{end_x} subvolume")
            except:
                print(f"Skipping z{start_z}-{end_z} y{start_y}-{end_y} x{start_x}-{end_x}")
            start_x = start_x + step_x
        start_y = start_y + step_y
    start_z = start_z + step_z