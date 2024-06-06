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
parser.add_argument('--img_dir', type=str, default="/home/hluo/scratch/results/savepred6", help='Image directory')
parser.add_argument('--save_dir', type=str, default="/home/hluo/scratch/pred6", help='Save directory')
parser.add_argument('--model_path', type=str, default="model_job84", help='Model name')
args = parser.parse_args()

# -- args
img_dir= args.img_dir
fp=args.model_path
save_dir=args.save_dir
if not os.path.exists(os.path.join(save_dir, "pred")): 
    os.makedirs(os.path.join(save_dir, "pred"), exist_ok=True)
    
# -- get imgs & mask fp
img_files = os.listdir(img_dir)
img_files = [os.path.join(img_dir, f) for f in img_files if f.endswith(".png")]

# -- load model
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model, optimizer, epoch, loss, batch_size, lr, focal_loss_weights = load_checkpoint(model, optimizer, fp)
model = model.eval()
print("model loaded")

for img_fp in img_files:
    img_name = os.path.basename(img_fp)
    img_name = os.path.splitext(img_name)[0]
    image = np.load(img_fp)
    image = torch.tensor(image).float().unsqueeze(0)
        
    try:
        image = tio.ZNormalization()(image)
        print(image.shape)
        intermed_pred, sub_volume_pred = model(image)
        binary_pred = torch.argmax(sub_volume_pred[0], dim=0)
        np.save(os.path.join(save_dir, "pred", f"{img_name}.npy"), sub_volume_pred.detach().cpu())            
    except:
        print(f"Skipping {img_name}")