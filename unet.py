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

model_folder = r"/home/mishaalk/scratch/gapjunc/models"
sample_preds_folder = r"/home/mishaalk/scratch/gapjunc/results"
table, class_labels = None, None #wandb stuff
                
def make_dataset_new(aug=False):
    x_new_dir = r"/home/mishaalk/scratch/gapjunc/seg_50_data/images"
    y_new_dir = r"/home/mishaalk/scratch/gapjunc/seg_50_data/labels"

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

def train_loop(model, train_data_loader, optimizer, val_dataloader=None, epochs=30, decay=None):
    global table, class_labels, model_folder, DEVICE
    
    print(f"Using device: {DEVICE}")
    model_name = "model5"
    def sigint_handler(sig, frame):
        if table is not None:
            print("logging to WANDB")
            wandb.log({"Table" : table})
            joblib.dump(model, os.path.join(model_folder, f"{model_name}.pk1"))
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            print("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, labels = data # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward() # calculate gradients (backpropagation)
            optimizer.step() # update model weights (values for kernels)
            print(f"Step: {i}, Loss: {loss}")
            loss_list.append(loss)
            wandb.log({"loss": loss})
        
        epoch_non_empty = False

        for i, data in enumerate(valid_loader):
            valid_inputs, valid_labels = data
            valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
            valid_pred = model(valid_inputs)
            valid_loss = criterion(valid_pred, valid_labels)
            mask_img = wandb.Image(valid_inputs[0].squeeze(0).cpu().numpy(), 
                                    masks = {
                                        "predictions" : {
                            "mask_data" : (torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))) * 255).cpu().detach().numpy(),
                            "class_labels" : class_labels
                        },
                        "ground_truth" : {
                            "mask_data" : (valid_labels[0].squeeze(0) * 255).cpu().numpy(),
                            "class_labels" : class_labels
                        }}
            )
            table.add_data(f"Epoch {epoch} Step {i}", mask_img)
            wandb.log({"valid_loss": valid_loss})
            uniques = np.unique(torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))).detach().cpu().numpy())
            if len(uniques) == 2:
                if not epoch_non_empty:
                    epoch_non_empty = True
                    print("UNIQUE OUTPUTS!")
            else:
                epoch_non_empty = False
        if decay is not None: decay.step(valid_loss)

        print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
        print(f"Time elapsed: {time.time() - start} seconds")
        temp_name = model_name+"_epoch"+str(epoch)
        joblib.dump(model, os.path.join(model_folder, f"{temp_name}.pk1"))
    print(f"Total time: {time.time() - start} seconds")
    wandb.log({"Table" : table})
    joblib.dump(model, os.path.join(model_folder, f"{model_name}.pk1"))
    wandb.finish()
    try:
        joblib.dump(loss_list, os.path.join(model_folder, "loss_list_1.pkl"))
    except:
        print("Failed to save loss list")

def inference_save(model, train_dataset, valid_dataset):
    global DEVICE, model_folder, sample_preds_folder

    sample_train_folder = sample_preds_folder+"//train_res"
    model = joblib.load(os.path.join(model_folder, "model5_epoch17.pk1"))
    model = model.to(DEVICE)
    model.eval()
    for i in tqdm(range(len(train_dataset))):
        image, gt_mask = train_dataset[i] # image and ground truth from test dataset
        # print(image.shape, gt_mask.shape) # [1, 512, 512] and [2, 512, 512]
        # print(image)
        suffix = "_1_{}".format(i)
        plt.imshow(image.squeeze(0).numpy(), cmap='gray')
        plt.savefig(os.path.join(sample_train_folder, f"sample_pred_{suffix}.png"))
        # plt.show()
        plt.imshow(gt_mask.squeeze(0).detach().numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_train_folder, f"sample_gt_{suffix}.png"))
        # plt.show()
        x_tensor = image.to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor) # [1, 2, 512, 512]
        # print(pred_mask.shape)
        # pred_mask_binary = pred_mask.squeeze(0).detach()
        pred_mask_binary = torch.round(nn.Sigmoid()(pred_mask)) * 255
        plt.imshow(pred_mask_binary.cpu().detach().squeeze(0).squeeze(0).numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_train_folder, f"sample_pred_binary_{suffix}.png"))
        # plt.show()

    sample_val_folder = sample_preds_folder+"//valid_res"
    for i in tqdm(range(len(valid_dataset))):
        image, gt_mask = valid_dataset[i] # image and ground truth from test dataset
        # print(image.shape, gt_mask.shape) # [1, 512, 512] and [2, 512, 512]
        # print(image)
        suffix = "_1_{}".format(i)
        plt.imshow(image.squeeze(0).numpy(), cmap='gray')
        plt.savefig(os.path.join(sample_val_folder, f"sample_pred_{suffix}.png"))
        # plt.show()
        plt.imshow(gt_mask.squeeze(0).detach().numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_val_folder, f"sample_gt_{suffix}.png"))
        # plt.show()
        x_tensor = image.to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor) # [1, 2, 512, 512]
        
        # pred_mask_binary = torch.argmax(pred_mask.squeeze(0).detach(), 0)
        pred_mask_binary = torch.round(nn.Sigmoid()(pred_mask)) * 255
        plt.imshow(pred_mask_binary.cpu().detach().squeeze(0).squeeze(0).numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_val_folder, f"sample_pred_binary_{suffix}.png"))


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
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--aug", action="store_true")
    parser.add_arguments("--new", action="store_true")
    parser.add_arguments("--model_name", default=None, type=str)


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

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model_name is None: model = UNet().to(DEVICE)
    else: model = joblib.load(os.path.join(model_folder, args.model_name)) # load model


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
    criterion = FocalLoss(alpha=cls_weights, device=DEVICE)#torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    decayed_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_list = [] 
    

    setup_wandb(30, 0.001)

    print("Starting training...")
    start = time.time()

    train_loop(model, train_loader, optimizer, valid_loader, epochs=30)

