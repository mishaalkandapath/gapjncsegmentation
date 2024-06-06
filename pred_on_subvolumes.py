import torchio as tio
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from models import *
from utilities import *
import gc

print("starting...")
parser = argparse.ArgumentParser(description="save preds")
parser.add_argument('--data_dir', type=str, default="/home/hluo/scratch/results/savepred6", help='Image directory')
parser.add_argument('--save_dir', type=str, default="/home/hluo/scratch/pred6", help='Save directory')
parser.add_argument('--model_path', type=str, default="model_job84", help='Model name')
args = parser.parse_args()

# -- args
data_dir= args.data_dir
img_dir = os.path.join(data_dir, "original")
mask_dir = os.path.join(data_dir, "ground_truth")
fp=args.model_path
save_dir=args.save_dir
if not os.path.exists(save_dir):
     os.makedirs(save_dir, exist_ok=True)
if not os.path.exists(os.path.join(save_dir, "pred")): 
    os.makedirs(os.path.join(save_dir, "pred"), exist_ok=True)
if not os.path.exists(os.path.join(save_dir, "visualize")): 
    os.makedirs(os.path.join(save_dir, "visualize"), exist_ok=True)
    
print("img dir", img_dir)
print("model", fp)
print("save dir", save_dir)
    
# -- get imgs & mask fp
img_files = os.listdir(img_dir)
img_files = [os.path.join(img_dir, f) for f in img_files if f.endswith(".npy")]
mask_files = os.listdir(mask_dir)
mask_files = [os.path.join(mask_dir, f) for f in mask_files if f.endswith(".npy")]
num_files = len(img_files)
print("Length of img_files", len(img_files), len(mask_files))

# -- load model
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model, optimizer, epoch, loss, batch_size, lr, focal_loss_weights = load_checkpoint(model, optimizer, fp)
model = model.eval()
print("model loaded")

start_time = time.time()
last_slice_time = time.time()
for i, img_fp in enumerate(img_files):
    img_name = os.path.basename(img_fp)
    img_name = os.path.splitext(img_name)[0]
    image = np.load(img_fp)
    if i == 0:
        depth = image.shape[0]
    mask = np.load(mask_files[i])
    image = torch.tensor(image).float().unsqueeze(0)
    slice_interval = time.time()-last_slice_time
    estim_time = (num_files-i)*(slice_interval)
    last_slice_time = time.time()
    print(f"Progress {i}/{num_files} {((i*100)/num_files):.2f}")
    print(image.shape, f"estimated time: {estim_time:.2f}s", f"slice time: {slice_interval:.2f}s")
    try:
        image = tio.ZNormalization()(image) # (channels=1, depth, height, width)
        sub_volume_pred = model(image)[1]
        print(f"pred time {time.time()-last_slice_time:.2f}")
        binary_pred = torch.argmax(sub_volume_pred[0], dim=0)
        np.save(os.path.join(save_dir, "pred", f"{img_name}.npy"), sub_volume_pred.detach().cpu()) 
        fig, ax = plt.subplots(3, depth, figsize=(15,5), num=1)
        visualize_3d_slice(image.detach().cpu()[0], ax[0], "Image")
        visualize_3d_slice(mask, ax[1], "Mask")
        visualize_3d_slice(binary_pred, ax[2], "Pred")
        plt.savefig(os.path.join(save_dir, "visualize", f"{img_name}.png"))
        plt.close("all")   
        del image, sub_volume_pred, binary_pred
        gc.collect()   
    except:
        print(f"Skipping {img_name}")

print("total time:", time.time()-start_time)