# import packages
import os
import cv2
import signal
import sys
import numpy as np;
import torch 
import torchvision.transforms as transforms
import wandb
import time
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utilities import *
from models import *
from dataset import *
from loss import *

def log_predictions(input_img, label_img, pred_img, epoch, step, table):
    """ 
    Log input, ground truth, and prediction images to wandb
    """
    depth, height, width = input_img.shape
    
    # -- plot as 3 rows: input, ground truth, prediction
    fig, ax = plt.subplots(3, depth, figsize=(15, 5), num=1)
    for j in range(depth):
        ax[0, j].imshow(input_img[j], cmap="gray")
        ax[1, j].imshow(label_img[j], cmap="gray")
        ax[2, j].imshow(pred_img[j], cmap="gray")
    ax[0, 0].set_ylabel("Input")
    ax[1, 0].set_ylabel("Ground Truth")
    ax[2, 0].set_ylabel("Prediction")
    
    # save figure to wandb
    mask_img = wandb.Image(fig)          
    table.add_data(f"Epoch {epoch} Step {step}", mask_img)

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, model_folder: str, model_name: str, results_dir:str, num_predictions_to_log:int=5) -> None:
    """ Train the model for a given number of epochs
    
    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        valid_loader (torch.utils.data.DataLoader): DataLoader for validation data
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epochs (int): number of epochs to train for
        model_folder (str): directory to save model checkpoints
        model_name (str): name of the model to save
        results_dir (str): directory to save results
    """
    depth, height, width = 5, 256, 256
    total_table = wandb.Table(columns=['Epoch', 'Image'])
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            print("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if (i == 0):
                print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
                print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
                _, _, depth, height, width = inputs.shape # initialize depth, height, width
            
            if inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {inputs.shape}")
                continue
            optimizer.zero_grad() # zero gradients (otherwise they accumulate)
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward() # calculate gradients
            optimizer.step() # update weights based on calculated gradients
            print(f"Step: {i}, Loss: {loss}")
            wandb.log({"loss": loss})
            
        # one artifact per epoch
        valid_artifact = wandb.Artifact(f"valid" + str(wandb.run.id), type="predictions")
        valid_table = wandb.Table(columns=['Epoch', 'Image'])
        num_logged = 0
        for i, data in enumerate(valid_loader):
            valid_inputs, valid_labels = data
            valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
            if valid_inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {valid_inputs.shape}")
                continue
            
            valid_pred = model(valid_inputs)
            valid_loss = criterion(valid_pred, valid_labels)
            print(f"Validation Step: {i}, input size: {valid_inputs.shape}, Loss: {valid_loss}")
            
            # Save predictions for each epoch
            if num_logged < num_predictions_to_log:
                print(f"Saving predictions for epoch {epoch} step {i}")
                input_img = valid_inputs.squeeze(0).squeeze(0).cpu().numpy()
                label_img = valid_labels[0][1].cpu().numpy()
                pred_img = np.argmax(valid_pred[0].detach().cpu(), 0).numpy()
                
                log_predictions(input_img, label_img, pred_img, epoch, i, valid_table) # adds row to artifact table
                log_predictions(input_img, label_img, pred_img, epoch, i, total_table) # adds row to total table
                num_logged += 1
                
            wandb.log({"valid_loss": valid_loss})
            plt.close("all")
        valid_artifact.add(valid_table, "predictions")
        wandb.run.log_artifact(valid_artifact)

        print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
        print(f"Time elapsed: {time.time() - start} seconds")
        checkpoint(model, optimizer, epoch, loss, batch_size, lr, inverse_class_freq, os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
    wandb.log({"Table" : total_table})
    wandb.finish()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a 3D U-Net model on the tiniest dataset")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--data_dir", type=str, default="data/tiniest_data_64", help="Directory containing the tiniest dataset")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--model_name", type=str, default="model1", help="Name of the model to save")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--gamma", type=float, default=3, help="Gamma parameter for Focal Loss")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha parameter for Focal Loss")
    parser.add_argument("--wandb_log_path", type=str, default="wandb", help="Path to save wandb logs")
    parser.add_argument("--num_predictions_to_log", type=int, default=5, help="Number of predictions to log per epoch")
    args = parser.parse_args()
    return args

def setup_datasets_and_dataloaders(data_dir, batch_size, num_workers, augment=False):
    x_train_dir = os.path.join(data_dir, "original", "train")
    y_train_dir = os.path.join(data_dir, "ground_truth", "train")
    x_valid_dir = os.path.join(data_dir, "original", "valid")
    y_valid_dir = os.path.join(data_dir, "ground_truth", "valid")
    
    train_dataset = SliceDataset(x_train_dir, y_train_dir, augment=augment)
    valid_dataset = SliceDataset(x_valid_dir, y_valid_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) # change num_workers as needed
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_dataset, valid_dataset, train_loader, valid_loader
    
    
if __name__ == "__main__":  
    # Define class labels (constant)
    class_labels = {
    0: "not gj",
    1: "gj",
    }
    
    # Check if GPU is available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
      
    # Parse arguments
    args = parse_arguments()
    
    # Define directories
    model_name = args.model_name
    model_folder = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_folder): os.makedirs(model_folder)
    data_dir = args.data_dir
    if not os.path.exists(data_dir): print(f"Data directory {data_dir} does not exist.")
    results_dir = os.path.join(args.results_dir, args.model_name)
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    # Get train and val dataset instances
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset, train_loader, valid_loader = setup_datasets_and_dataloaders(data_dir, batch_size, num_workers)
    print(f"Data loaders created. Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")
    print(f"Batch size: {batch_size}, Number of workers: {num_workers}")

    # Initialize model
    lr = args.lr
    epochs = args.epochs
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    load_model_path = args.load_model_path
    if load_model_path is not None:
        model, optimizer, start_epoch, loss, batch_size, lr, focal_loss_weights = load_checkpoint(model, optimizer, load_model_path)
    model = model.to(DEVICE)
    print(f"Model is on device {next(model.parameters()).device}")

    # Initialize loss function
    print("Calculating alpha values for focal loss...")
    if args.alpha is None:
        inverse_class_freq = get_inverse_class_frequencies(train_dataset)
        alpha = torch.Tensor(inverse_class_freq)
        alpha = scale_to_sum_to_one(alpha).to(DEVICE)
    else:
        alpha = torch.Tensor([args.alpha, 1-args.alpha]).to(DEVICE)
    print(f"Alpha values: {alpha}")
    gamma = args.gamma
    criterion = FocalLoss(alpha=alpha, gamma=gamma, device=DEVICE)
    print("Loss function initialized.")
    
    # Initialize wandb
    wandb.init(
        project="gapjnc-dense-cell",
        config={
        "learning_rate": lr,
        "epochs": epochs,
        "alpha": alpha,
        "model_name": model_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "inverse class frequencies": inverse_class_freq,
        "gamma": gamma
        },
        dir=args.wandb_log_path
    )      

    # Train model
    print("Starting training...")
    start = time.time()
    train(model, train_loader, valid_loader, criterion, optimizer, epochs, model_folder, model_name, results_dir, args.num_predictions_to_log)
    print(f"Training complete. Time elapsed: {time.time() - start} seconds")
