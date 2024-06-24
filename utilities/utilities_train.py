"""
Utility functions for training the model
"""
import os
import torch 
import wandb
import time
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utilities.utilities import *
from utilities.models import *
from utilities.dataset import *
from utilities.loss import *

def parse_arguments():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description="Train a 3D U-Net model on the tiniest dataset")
    
    # directory arguments
    parser.add_argument("--train_x_dirs", type=str, nargs='+', default=None, help="Directories to get original images from")
    parser.add_argument("--train_y_dirs", type=str, nargs='+', default=None, help="Directories to get mask images from")
    parser.add_argument("--valid_x_dirs", type=str, nargs='+', default=None, help="Directories to get original images from")
    parser.add_argument("--valid_y_dirs", type=str, nargs='+', default=None, help="Directories to get mask images from")
    parser.add_argument("--train_cellmask_dir", type=str, default=None, help="Path to load cellmask from")
    parser.add_argument("--valid_cellmask_dir", type=str, default=None, help="Path to load cellmask from")
    parser.add_argument("--loss_dir", type=str, help="Directory to save loss")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--wandb_log_path", type=str, default="wandb", help="Path to save wandb logs")
    
    # model & training arguments
    parser.add_argument("--model_name", type=str, default="model1", help="Name of the model to save")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--freeze_model_start_layer", type=int, default=None, help="Layer to start unfreezing model from")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--num_predictions_to_log", type=int, default=5, help="Number of predictions to log per epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # loss arguments
    parser.add_argument("--pred3classes", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to predict 3 classes")
    parser.add_argument("--augment", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to augment the data")
    parser.add_argument("--loss_type", type=str, default="focal", help="Type of loss function to use")
    parser.add_argument("--use_dice", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to use Dice loss")
    parser.add_argument("--gamma", type=float, default=3, help="Gamma parameter for Focal Loss")
    parser.add_argument("--alpha", type=float, default=0.04, help="Weight for class 0 in Focal Loss")
    parser.add_argument("--beta", type=float, default=0.96, help="Weight for class 0 in Focal Loss")
    parser.add_argument("--ce_ratio", type=float, default=0.5, help="Weight in Combo Loss")
    
    # input arguments
    parser.add_argument("--downsample_factor", type=int, default=1, help="factor to downsample the data by")
    parser.add_argument("--height", type=int, default=512, help="Height of input")
    parser.add_argument("--width", type=int, default=512, help="Width of input")
    parser.add_argument("--depth", type=int, default=3, help="Depth of input")

    args = parser.parse_args()
    return args

def log_predictions(input_img: np.ndarray, label_img: np.ndarray, pred_img: np.ndarray, epoch: int, step: int, table: wandb.Table):
    """ Log input, ground truth, and prediction images to wandb """
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

def setup_datasets_and_dataloaders_from_lists(img_dir_list, mask_dir_list, batch_size: int, num_workers: int, augment: bool=True, shuffle: bool=True, downsample_factor: int=1):
    """ Setup datasets and dataloaders for training and validation"""
    print("Setting up: augment ", augment, " shuffle ", shuffle)
    my_dataset = SliceDatasetMultipleFolders(img_dir_list, mask_dir_list, augment=augment, downsample_factor=downsample_factor)
    my_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) # change num_workers as needed
    return my_dataset, my_loader

def setup_datasets_and_dataloaders(x_dir: str, y_dir: str, batch_size: int, num_workers: int, augment: bool=False, shuffle: bool=True, downsample_factor: int=1):
    """ Setup datasets and dataloaders for training and validation"""
    my_dataset = SliceDataset(x_dir, y_dir, augment=augment, downsample_factor=downsample_factor)
    my_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) # change num_workers as needed
    return my_dataset, my_loader

def setup_datasets_and_dataloaders_withmemb(x_dir: str, y_dir:str, cellmask_dir:str, batch_size: int, num_workers: int, augment: bool=False, use3classes: bool=False, shuffle: bool=True, downsample_factor: int=1):
    """ Setup datasets and dataloaders for training and validation"""
    if use3classes:
        my_dataset = SliceDatasetWithMembThreeClasses(x_dir, y_dir, cellmask_dir, augment=augment, downsample_factor=1)
    else:
        my_dataset = SliceDatasetWithMemb(x_dir, y_dir, cellmask_dir, augment=augment, downsample_factor=1)
    my_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) # change num_workers as needed
    return my_dataset, my_loader

def freeze_model_layers(model, freeze_model_start_layer):
    if freeze_model_start_layer is not None:
        # first freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        print("Freezed all layers")
            
        # then unfreeze small portion
        if freeze_model_start_layer > 0:
            # -- last layer only --
            for param in model.single_conv3.parameters():
                param.requires_grad = True
            print("unfreezed layer 1")
            if freeze_model_start_layer > 1:
                # -- pyramid1 -- 
                for param in model.single_three_conv1.parameters():
                    param.requires_grad = True
                for param in model.pyramid1.parameters():
                    param.requires_grad = True
                for param in model.res_conn.parameters():
                    param.requires_grad = True
                print("unfreezed layer 2 (short pyramid)")
                # -- pyramid2 -- 
                if freeze_model_start_layer > 2:
                    for param in model.pyramid2.parameters():
                        param.requires_grad = True
                    for param in model.single_three_conv2.parameters():
                        param.requires_grad = True
                    for param in model.single_conv2.parameters():
                        param.requires_grad = True
                    for param in model.transpose.parameters():
                        param.requires_grad = True
                    print("unfreezed layer 3 (long pyramid)")
                    if freeze_model_start_layer > 3:
                        for param in model.one_conv1.parameters():
                            param.requires_grad = True
                        for param in model.up_conv1.parameters():
                            param.requires_grad = True
                        for param in model.up_conv2.parameters():
                            param.requires_grad = True
                        for param in model.up_conv3.parameters():
                            param.requires_grad = True
                        print("unfreezed layer 4 (upsampling path)")
    return model

def train_log_local(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,lr: float,model_folder: str, model_name: str, results_folder:str, loss_folder:str, num_predictions_to_log:int=5, depth=3, height=512, width=512, use2d3d=False, savevis=True) -> None:
    """ 
    Train the model and log predictions locally.
    
    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        valid_loader (torch.utils.data.DataLoader): DataLoader for validation data
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epochs (int): number of epochs to train for
        batch_size (int): batch size for training
        lr (float): learning rate
        model_folder (str): directory to save model checkpoints
        model_name (str): name of the model to save
        num_predictions_to_log (int): number of predictions to log per epoch
    """
    start = time.time()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses = []
    train_precision = []
    train_recall = []
    valid_losses = []
    valid_precision = []
    valid_recall = []
    
    # --- Training ---
    for epoch in range(epochs):
        print(f"--------------------Epoch {epoch} (time: {time.time() - start} seconds)--------------------")
        epoch_train_precision = 0
        epoch_train_recall = 0
        epoch_train_loss = 0
        num_train_logged = 0
        num_train_processed = 1
        epoch_tp=0
        epoch_fp=0
        epoch_tn=0
        epoch_fn=0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            print("Progress: {:.2%}".format(i/len(train_loader)), end="\r")
            if (i==0 and epoch==0):
                print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}") # (batch, channel, depth, height, width)
                print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
            if inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {inputs.shape}")
                continue
            
            # Calculate loss and update weights
            optimizer.zero_grad()
            intermediate_pred, pred = model(inputs)      
            if use2d3d:
                loss = criterion(intermediate_pred, pred, labels)
            else:
                loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            num_train_processed += 1
            epoch_train_loss += loss.detach().cpu().item()
            
            # Get metrics
            mask_for_metric = labels[:, 1]
            pred_for_metric = torch.argmax(pred, dim=1) 
            tp, fp, fn, tn = get_confusion_matrix(pred=pred_for_metric, target=mask_for_metric)
            epoch_tp += tp
            epoch_fp += fp
            epoch_fn += fn
            epoch_tn += tn

            # Save predictions for each epoch
            if num_train_logged < num_predictions_to_log:
                input_img = inputs[0][0].cpu().numpy()
                label_img = labels[0][1].cpu().numpy()
                pred_img = np.argmax(pred[0].detach().cpu(), 0).numpy()
                fig, ax = plt.subplots(3, depth, figsize=(15, 5), num=1)
                for j in range(depth):
                    ax[0, j].imshow(input_img[j], cmap="gray")
                    ax[1, j].imshow(label_img[j], cmap="gray")
                    ax[2, j].imshow(pred_img[j], cmap="gray")
                ax[0, 0].set_ylabel("Input")
                ax[1, 0].set_ylabel("Ground Truth")
                ax[2, 0].set_ylabel("Prediction")
                plt.savefig(os.path.join(results_folder, "train", f"epoch{epoch}_num{num_train_logged}.png"))
                num_train_logged += 1
                plt.close("all")
        
        # save metrics
        epoch_train_loss = epoch_train_loss/(num_train_processed)
        epoch_train_precision = epoch_tp/(epoch_tp+epoch_fp) if (epoch_tp+epoch_fp) > 0 else 0
        epoch_train_recall = epoch_tp/(epoch_tp+epoch_fn) if (epoch_tp+epoch_fn) > 0 else 0
        train_losses.append(epoch_train_loss)
        train_precision.append(epoch_train_precision)
        train_recall.append(epoch_train_recall)
        print(f"Train | loss: {epoch_train_loss:.5f} precision: {epoch_train_precision:.5f} recall: {epoch_train_recall:.5f}")
        
        # --- Validation ---
        epoch_valid_precision = 0
        epoch_valid_recall = 0
        epoch_valid_loss = 0
        epoch_tn=0
        epoch_tp=0
        epoch_fn=0
        epoch_fp=0
        num_logged = 0
        num_valid_processed = 1
        for i, data in enumerate(valid_loader):
            valid_inputs, valid_labels = data
            valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
            if valid_inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {valid_inputs.shape}")
                continue
            
            # calculate loss
            valid_interm_pred, valid_pred = model(valid_inputs)
            if use2d3d:
                valid_loss = criterion(valid_interm_pred,valid_pred, valid_labels)
            else:
                valid_loss = criterion(valid_pred, valid_labels)
            
            # log metrics
            num_valid_processed += 1
            epoch_valid_loss += valid_loss.detach().cpu().item()
            mask_for_metric = valid_labels[:, 1]
            pred_for_metric = torch.argmax(valid_pred, dim=1) 
            tp, fp, fn, tn = get_confusion_matrix(pred=pred_for_metric, target=mask_for_metric)
            epoch_tp += tp
            epoch_fp += fp
            epoch_fn += fn
            epoch_tn += tn

            # Save predictions
            if num_logged < num_predictions_to_log:
                input_img = valid_inputs[0][0].cpu().numpy()
                label_img = valid_labels[0][1].cpu().numpy()
                pred_img = np.argmax(valid_pred[0].detach().cpu(), 0).numpy()
                fig, ax = plt.subplots(3, depth, figsize=(15, 5), num=1)
                for j in range(depth):
                    ax[0, j].imshow(input_img[j], cmap="gray")
                    ax[1, j].imshow(label_img[j], cmap="gray")
                    ax[2, j].imshow(pred_img[j], cmap="gray")
                ax[0, 0].set_ylabel("Input")
                ax[1, 0].set_ylabel("Ground Truth")
                ax[2, 0].set_ylabel("Prediction")
                plt.savefig(os.path.join(results_folder, "valid", f"epoch{epoch}_num{num_logged}.png"))
                num_logged += 1
            valid_losses.append(valid_loss.detach().cpu().item())
            plt.close("all")
            
        # save metrics
        epoch_valid_loss = epoch_valid_loss/(num_valid_processed)
        epoch_valid_precision = epoch_tp/(epoch_tp+epoch_fp) if (epoch_tp+epoch_fp) > 0 else 0
        epoch_valid_recall = epoch_tp/(epoch_tp+epoch_fn) if (epoch_tp+epoch_fn) > 0 else 0
        valid_precision.append(epoch_valid_precision)
        valid_recall.append(epoch_valid_recall)
        valid_losses.append(epoch_valid_loss)
        print(f"Valid | loss: {epoch_valid_loss:.5f} precision: {epoch_valid_precision:.5f} recall: {epoch_valid_recall:.5f}")
        
        checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss, batch_size=batch_size, lr=lr, focal_loss_weights=(0, 0), path=os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
        torch.save({
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'valid_precision': valid_precision,
            'valid_recall': valid_recall,
        }, os.path.join(results_folder, f"losses_{model_name}.pth"))
        
        if savevis:
            fig, ax= plt.subplots(2,1,figsize=(10, 8))
            ax[0].set_title("Train Loss")
            ax[0].plot(train_losses)
            ax[0].set_xticks(np.arange(0,  100, 5))
            ax[1].set_title("Valid Loss")
            ax[1].plot(valid_losses)
            ax[1].set_xticks(np.arange(0,  100, 5))
            plt.suptitle(f"Model {model_name}")
            plt.savefig(f"losses_{model_name}.png")
    print(f"Training complete. Time elapsed: {time.time() - start} seconds")

