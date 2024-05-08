# import packages
import os
import cv2
import numpy as np;
import random, tqdm
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# import segmentation_models_pytorch as smp
# import albumentations as album
import joblib
import torchvision.ops.focal_loss as focal
from torchvision.transforms import v2

from typing import Tuple, List
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from utilities import *
# import segmentation_models_pytorch.utils.metrics
import wandb
import random
import torch
import time
from tqdm import tqdm
import signal
import sys
import math

model_folder = r"/home/mishaalk/scratch/gapjunc/models"
sample_preds_folder = r"/home/mishaalk/scratch/gapjunc/results"
table, class_labels = None, None #wandb stuff
                
def make_dataset_new(x_new_dir, y_new_dir, aug=False):
    height, width = cv2.imread(os.path.join(x_new_dir, os.listdir(x_new_dir)[0])).shape[:2]

    # Get train and val dataset instances
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.4)
    ])

    dataset = CaImagesDataset(
        x_new_dir, y_new_dir, 
        preprocessing=None,
        image_dim = (width, height), augmentation=augmentation if aug else None
    )

    train, valid = torch.utils.random_split(dataset, [len(dataset)-200, 200])
    return train, valid

def make_dataset_old(aug=False):

    x_train_dir=r"/home/mishaalk/scratch/gapjunc/small_data/original/train"
    y_train_dir=r"/home/mishaalk/scratch/gapjunc/small_data/ground_truth/train"

    x_valid_dir=r"/home/mishaalk/scratch/gapjunc/small_data/original/valid"
    y_valid_dir=r"/home/mishaalk/scratch/gapjunc/small_data/ground_truth/valid"

    x_test_dir=r"/home/mishaalk/scratch/gapjunc/small_data/original/test"
    y_test_dir=r"/home/mishaalk/scratch/gapjunc/small_data/ground_truth/test"

    # Get train and val dataset instances
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.4)
    ])

    height, width = cv2.imread(os.path.join(x_train_dir, os.listdir(x_train_dir)[0])).shape[:2]

    train_dataset = CaImagesDataset(
        x_train_dir, y_train_dir, 
        preprocessing=None,
        image_dim = (width, height), augmentation=augmentation if aug else None
    )

    valid_dataset = CaImagesDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=None,
        preprocessing=None,
        image_dim = (width, height)
    )

    return train_dataset, valid_dataset




def setup_wandb(epochs, lr):
    global table, class_labels
    WANDB_API_KEY = "42a2147c44b602654473783bde1ecd15579cc313"
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="celegans",
        entity="mishaalkandapath",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": epochs,
        }
    )
    class_labels = {
        1: "background",
        0: "gapjunction",
    }

    table = wandb.Table(columns=['ID', 'Image'])


if __name__ == "__main__":
    import argparser

    parser = argparser.ArgumentParser()
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--aug", action="store_true")
    parser.add_arguments("--new", action="store_true")
    parser.add_arguments("--model_name", default=None, type=str)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--cpu", action="store_true")


    args = parser.parse_args()

    if args.seed:
        SEED = 12
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.manual_seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)

    batch_size = args.batch_size

    train_dataset, valid_dataset = make_dataset_new(args.aug) if args.new else make_dataset_old(args.aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)
    print("Data loaders created.")

    DEVICE = torch.device("cuda") if torch.cuda.is_available() or not args.cpu else torch.device("cpu")

    if args.model_name is None: model = SplitUNet().to(DEVICE)
    else: model = joblib.load(os.path.join(model_folder, args.model_name)) # load model

    if not args.infer:
        #calc focal weighting:
        smushed_labels = None
        for i in range(len(train_dataset)):
            if smushed_labels is None: smushed_labels = train_dataset[i][1].to(torch.int64)
            else: smushed_labels = torch.concat([smushed_labels, train_dataset[i][1].to(torch.int64)])
        class_counts = torch.bincount(smushed_labels.flatten())
        total_samples = len(train_dataset) * 512 * 512
        w1, w2 = 1/(class_counts[0]/total_samples), 1/(class_counts[1]/total_samples)
        cls_weights = torch.Tensor([w1, w2/9])
        print(cls_weights)

        #init oprtimizers
        criterion = FocalLoss(alpha=cls_weights, device=DEVICE)
        classifier_criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        decayed_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        loss_list = [] 
        

        setup_wandb(30, 0.001)

        print("Starting training...")
        start = time.time()

        train_loop(model, train_loader, criterion, classifier_criterion optimizer, valid_loader, epochs=30)
    else:
        inference_save(model, train_dataset, valid_dataset)

