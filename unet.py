# import packages
import os
import cv2
import numpy as np;
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import wandb
import random
import time

from scipy import ndimage
from typing import Tuple, List
from scipy import stats
from torch.utils.data import DataLoader
from utilities import *
from models import *


# Define directories
x_train_dir=r"C:\Users\hclov\Documents\code\gap-junction\data_small\original\train"
y_train_dir=r"C:\Users\hclov\Documents\code\gap-junction\data_small\ground_truth\train"

x_valid_dir=r"C:\Users\hclov\Documents\code\gap-junction\data_small\original\valid"
y_valid_dir=r"C:\Users\hclov\Documents\code\gap-junction\data_small\ground_truth\valid"

x_test_dir=r"C:\Users\hclov\Documents\code\gap-junction\data_small\original\test"
y_test_dir=r"C:\Users\hclov\Documents\code\gap-junction\data_small\ground_truth\test"

model_folder = r"C:\Users\hclov\Documents\code\gap-junction\models"
sample_preds_folder = r"C:\Users\hclov\Documents\code\gap-junction\results"

height, width = cv2.imread(os.path.join(x_train_dir, os.listdir(x_train_dir)[0])).shape[:2]

class_labels = {
0: "background",
1: "calcium",
}

# Get train and val dataset instances
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # flip with 50% probability
    transforms.RandomVerticalFlip(p=0.5), # flip with 50% probability
    transforms.RandomRotation(degrees=90), # rotate with 90 degrees
])
train_dataset = SliceDataset(x_train_dir, y_train_dir, image_dim = (width, height), augmentation=augmentation)
valid_dataset = SliceDataset(x_valid_dir, y_valid_dir, image_dim = (width, height))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
print("Data loaders created.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Training parameters
lr = 0.001
epochs = 20      

# Initialize model
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
load_model_path = None
if load_model_path is not None:
    model, optimizer, start_epoch, loss = load_checkpoint(model, os.path.join(model_folder, "model1.pk1"))
model.train()

# Initialize loss function
smushed_labels = None
for i in range(len(train_dataset)):
    if smushed_labels is None: smushed_labels = train_dataset[i][1].to(torch.int64)
    else: smushed_labels = torch.concat([smushed_labels, train_dataset[i][1].to(torch.int64)])
class_counts = torch.bincount(smushed_labels.flatten())
total_samples = len(train_dataset) * 512 * 512
w1, w2 = 1/(class_counts[0]/total_samples), 1/(class_counts[1]/total_samples)
alpha = torch.Tensor([w1, w2/9])
gamma = 3
criterion = FocalLoss(alpha=alpha, gamma=gamma, device=DEVICE)

wandb.init(
    project="gap-junction",
    config={
    "learning_rate": lr,
    "epochs": epochs,
    "alpha": alpha,
    }
)      
table = wandb.Table(columns=['ID', 'Image'])


# Train model
print("Starting training...")
start = time.time()
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        print("Progress: {:.2%}".format(i/len(train_loader)))
        inputs, labels = data
        optimizer.zero_grad() # zero gradients (otherwise they accumulate)
        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward() # calculate gradients
        optimizer.step() # update weights based on calculated gradients
        print(f"Step: {i}, Loss: {loss}")
        wandb.log({"loss": loss})

    for i, data in enumerate(valid_loader):
        valid_inputs, valid_labels = data
        valid_pred = model(valid_inputs)
        valid_loss = criterion(valid_pred, valid_labels)
        mask_img = wandb.Image(
            valid_inputs[0].squeeze(0).numpy(), # original image
            masks = {
                "predictions" : {"mask_data" : np.argmax(valid_pred[0].detach(), 0).numpy(), "class_labels" : class_labels},
                "ground_truth" : {"mask_data" : valid_labels[0][1].numpy(),"class_labels" : class_labels}
            }
            )
        table.add_data(f"Epoch {epoch} Step {i}", mask_img)
        wandb.log({"valid_loss": valid_loss})

    print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
    print(f"Time elapsed: {time.time() - start} seconds")
    checkpoint(model, optimizer, epoch, loss, os.path.join(model_folder, f"model1_epoch{epoch}.pk1"))
print(f"Total time: {time.time() - start} seconds")
wandb.log({"Table" : table})
checkpoint(model, optimizer, epoch, loss, os.path.join(model_folder, "model1.pk1"))
wandb.finish()
