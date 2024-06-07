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
from utilities import *
from models import *
from dataset import *
from loss import *

def parse_arguments():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description="Train a 3D U-Net model on the tiniest dataset")
    parser.add_argument("--data_dir", type=str, default="data/tiniest_data_64", help="Directory containing the tiniest dataset")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="model1", help="Name of the model to save")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--gamma", type=float, default=3, help="Gamma parameter for Focal Loss")
    parser.add_argument("--alpha", type=float, default=0.04, help="Weight for class 0 in Focal Loss")
    parser.add_argument("--beta", type=float, default=0.96, help="Weight for class 0 in Focal Loss")
    parser.add_argument("--ce_ratio", type=float, default=0.5, help="Weight in Combo Loss")
    parser.add_argument("--wandb_log_path", type=str, default="wandb", help="Path to save wandb logs")
    parser.add_argument("--num_predictions_to_log", type=int, default=5, help="Number of predictions to log per epoch")
    parser.add_argument("--augment", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to augment the data")
    parser.add_argument("--loss_type", type=str, default="focal", help="Type of loss function to use")
    parser.add_argument("--use_dice", type=lambda x: (str(x).lower() == 'true'), default=False, help="Type of loss function to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

def setup_datasets_and_dataloaders(data_dir: str, batch_size: int, num_workers: int, augment: bool=False):
    """ Setup datasets and dataloaders for training and validation"""
    x_train_dir = os.path.join(data_dir, "original", "train")
    y_train_dir = os.path.join(data_dir, "ground_truth", "train")
    x_valid_dir = os.path.join(data_dir, "original", "valid")
    y_valid_dir = os.path.join(data_dir, "ground_truth", "valid")
    
    train_dataset = SliceDataset(x_train_dir, y_train_dir, augment=augment)
    valid_dataset = SliceDataset(x_valid_dir, y_valid_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) # change num_workers as needed
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_dataset, valid_dataset, train_loader, valid_loader

def train_log_local(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,lr: float,model_folder: str, model_name: str, results_folder:str, num_predictions_to_log:int=5) -> None:
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
    valid_losses = []
    train_tp = []
    train_fp = []
    train_fn = []
    train_tn = []
    valid_tp = []
    valid_fp = []
    valid_fn = []
    valid_tn = []
    train_precision = []
    train_recall = []
    valid_precision = []
    valid_recall = []
    epoch_train_precisions = []
    epoch_valid_precisions = []
    epoch_train_recalls = []
    epoch_valid_recalls = []
    first_img=True
    for epoch in range(epochs):
        num_train_logged = 0
        epoch_valid_precision = 0
        epoch_train_precision = 0
        epoch_valid_recall = 0
        epoch_train_recall = 0
        for i, data in enumerate(train_loader):
            print("Progress: {:.2%}".format(i/len(train_loader)), end="\r")
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if (first_img):
                print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}") # (batch, channel, depth, height, width)
                print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
                _, _, depth, height, width = inputs.shape # initialize depth, height, width
                if height != width:
                    continue
                else:
                    first_img=False
                    print(f"depth {depth}, height {height}, width {width}")
            
            if inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {inputs.shape}")
                continue
            optimizer.zero_grad() # zero gradients (otherwise they accumulate)
            intermediate_pred, pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward() # calculate gradients
            optimizer.step() # update weights based on calculated gradients
            mask_for_metric = labels[:, 1]
            pred_for_metric = torch.argmax(pred, dim=1) 
            accuracy = get_accuracy(pred=pred_for_metric, target=mask_for_metric)
            precision = get_precision(pred=pred_for_metric, target=mask_for_metric)
            epoch_train_precision += precision
            recall = get_recall(pred=pred_for_metric, target=mask_for_metric)
            epoch_train_recall += recall
            tp, fp, fn, tn = get_confusion_matrix(pred=pred_for_metric, target=mask_for_metric)
            print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
            print(f"TP: {tp}, TN: {tn} | FP: {fp}, FN: {fn}")
            # print(f"Step: {i}, Loss: {loss}")
            train_tn.append(tn)
            train_tp.append(tp)
            train_fn.append(fn)
            train_fp.append(fp)
            train_precision.append(precision)
            train_recall.append(recall)
            train_losses.append(loss.detach().cpu().item())

            # Save predictions for each epoch
            if num_train_logged < num_predictions_to_log:
                # input_img = inputs.squeeze(0).squeeze(0).cpu().numpy()
                input_img = inputs[0][0].cpu().numpy()
                label_img = labels[0][1].cpu().numpy()
                pred_img = np.argmax(pred[0].detach().cpu(), 0).numpy()
                
                # -- plot as 3 rows: input, ground truth, prediction
                fig, ax = plt.subplots(3, depth, figsize=(15, 5), num=1)
                for j in range(depth):
                    ax[0, j].imshow(input_img[j], cmap="gray")
                    ax[1, j].imshow(label_img[j], cmap="gray")
                    ax[2, j].imshow(pred_img[j], cmap="gray")
                ax[0, 0].set_ylabel("Input")
                ax[1, 0].set_ylabel("Ground Truth")
                ax[2, 0].set_ylabel("Prediction")
                plt.savefig(os.path.join(results_folder, "train", f"num{num_train_logged}_epoch{epoch}.png"))
                num_train_logged += 1
            plt.close("all")
        print(f"Epoch: {epoch}, Loss: {loss}")
        num_logged = 0
        for i, data in enumerate(valid_loader):
            valid_inputs, valid_labels = data
            valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
            if valid_inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {valid_inputs.shape}")
                continue
            
            valid_interm_pred, valid_pred = model(valid_inputs)
            valid_loss = criterion(valid_pred, valid_labels)
            mask_for_metric = valid_labels[:, 1]
            pred_for_metric = torch.argmax(valid_pred, dim=1) 
            accuracy = get_accuracy(pred=pred_for_metric, target=mask_for_metric)
            precision = get_precision(pred=pred_for_metric, target=mask_for_metric)
            recall = get_recall(pred=pred_for_metric, target=mask_for_metric)
            epoch_valid_precision += precision
            epoch_valid_recall += recall
            valid_precision.append(precision)
            valid_recall.append(recall)
            tp, fp, fn, tn = get_confusion_matrix(pred=pred_for_metric, target=mask_for_metric)
            print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
            print(f"TP: {tp}, TN: {tn} | FP: {fp}, FN: {fn}")
            # print(f"Step: {i}, Loss: {loss}")
            valid_tn.append(tn)
            valid_tp.append(tp)
            valid_fn.append(fn)
            valid_fp.append(fp)
            # Save predictions for each epoch
            if num_logged < num_predictions_to_log:
                # input_img = valid_inputs.squeeze(0).squeeze(0).cpu().numpy()
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
        try:
            print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
        except:
            print(f"Epoch: {epoch} | Loss: {loss}")
        print(f"Time elapsed: {time.time() - start} seconds")
        try:
            checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss, batch_size=batch_size, lr=lr, focal_loss_weights=(criterion.gamma, criterion.alpha), path=os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
        except:
            checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss, batch_size=batch_size, lr=lr, focal_loss_weights=(0, 0), path=os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
        torch.save({
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'valid_precision': valid_precision,
            'valid_recall': valid_recall,
            'train_tp': train_tp,
            'train_fn': train_fn,
            'train_fp': train_fp,
            'train_tn': train_tn,
            'valid_tp': valid_tp,
            'valid_fn': valid_fn,
            'valid_fp': valid_fp,
            'valid_tn': valid_tn,
        }, os.path.join(results_folder, "losses.pth"))
    print(f"Training complete. Time elapsed: {time.time() - start} seconds")

def train_log_local_2d3d(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,lr: float,model_folder: str, model_name: str, results_folder:str, num_predictions_to_log:int=5) -> None:
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
    valid_losses = []
    first_img=True
    for epoch in range(epochs):
        num_train_logged = 0
        for i, data in enumerate(train_loader):
            print("Progress: {:.2%}".format(i/len(train_loader)), end="\r")
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if (first_img):
                print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
                print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
                _, _, depth, height, width = inputs.shape # initialize depth, height, width
                if height != width:
                    continue
                else:
                    first_img=False
                    print(f"depth {depth}, height {height}, width {width}")
            
            if inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {inputs.shape}")
                continue
            optimizer.zero_grad() # zero gradients (otherwise they accumulate)
            intermediate_pred, pred = model(inputs)
            loss = criterion(intermediate_pred, pred, labels)
            loss.backward() # calculate gradients
            optimizer.step() # update weights based on calculated gradients
            # print(f"Step: {i}, Loss: {loss}")
            train_losses.append(loss.detach().cpu().item())

            # Save predictions for each epoch
            if num_train_logged < num_predictions_to_log:
                # input_img = inputs.squeeze(0).squeeze(0).cpu().numpy()
                input_img = inputs[0][0].cpu().numpy()
                label_img = labels[0][1].cpu().numpy()
                pred_img = np.argmax(pred[0].detach().cpu(), 0).numpy()
                
                # -- plot as 3 rows: input, ground truth, prediction
                fig, ax = plt.subplots(3, depth, figsize=(15, 5), num=1)
                for j in range(depth):
                    ax[0, j].imshow(input_img[j], cmap="gray")
                    ax[1, j].imshow(label_img[j], cmap="gray")
                    ax[2, j].imshow(pred_img[j], cmap="gray")
                ax[0, 0].set_ylabel("Input")
                ax[1, 0].set_ylabel("Ground Truth")
                ax[2, 0].set_ylabel("Prediction")
                plt.savefig(os.path.join(results_folder, "train", f"num{num_train_logged}_epoch{epoch}.png"))
                num_train_logged += 1
            plt.close("all")
        print(f"Epoch: {epoch}, Loss: {loss}")
            
        num_logged = 0
        for i, data in enumerate(valid_loader):
            valid_inputs, valid_labels = data
            valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
            if valid_inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {valid_inputs.shape}")
                continue
            
            valid_interm_pred, valid_pred = model(valid_inputs)
            valid_loss = criterion(valid_interm_pred, valid_pred, valid_labels)
            # Save predictions for each epoch
            if num_logged < num_predictions_to_log:
                # input_img = valid_inputs.squeeze(0).squeeze(0).cpu().numpy()
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
        try:
            print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
        except:
            print(f"Epoch: {epoch} | Loss: {loss}")
        print(f"Time elapsed: {time.time() - start} seconds")
        try:
            checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss, batch_size=batch_size, lr=lr, focal_loss_weights=(criterion.gamma, criterion.alpha), path=os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
        except:
            checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss, batch_size=batch_size, lr=lr, focal_loss_weights=(0, 0), path=os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
        torch.save({
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }, os.path.join(results_folder, "losses.pth"))
    print(f"Training complete. Time elapsed: {time.time() - start} seconds")

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,lr: float,model_folder: str, model_name: str, num_predictions_to_log:int=5) -> None:
    """ 
    Train the model and log predictions to wandb.
    
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
    total_table = wandb.Table(columns=['Epoch', 'Image'])
    total_train_table = wandb.Table(columns=['Epoch', 'train_image'])
    start = time.time()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        train_artifact = wandb.Artifact(f"train" + str(wandb.run.id), type="predictions")
        train_table = wandb.Table(columns=['Epoch', 'Image'])
        num_train_logged = 0
        for i, data in enumerate(train_loader):
            print("Progress: {:.2%}".format(i/len(train_loader)), end="\r")
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if (i == 0):
                print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
                print(f"Inputs device: {inputs.device}, Labels device: {labels.device}")
                # print(f"Unique labels: {np.unique(labels.detach().numpy(), return_counts=True)}")
                _, _, depth, height, width = inputs.shape # initialize depth, height, width
                print(f"depth {depth}, height {height}, width {width}")
            
            if inputs.shape[2:] != (depth, height, width):
                print(f"Skipping batch {i} due to shape mismatch, input shape: {inputs.shape}")
                continue
            optimizer.zero_grad() # zero gradients (otherwise they accumulate)
            intermediate_pred, pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward() # calculate gradients
            optimizer.step() # update weights based on calculated gradients
            # print(f"Step: {i}, Loss: {loss}")
            wandb.log({"loss": loss})

            # Save predictions for each epoch
            if num_train_logged < num_predictions_to_log:
                input_img = inputs.squeeze(0).squeeze(0).cpu().numpy()
                label_img = labels[0][1].cpu().numpy()
                pred_img = np.argmax(pred[0].detach().cpu(), 0).numpy()
                # pred_img = pred[0].detach().cpu().numpy()
                
                log_predictions(input_img, label_img, pred_img, epoch, i, train_table)
                log_predictions(input_img, label_img, pred_img, epoch, i, total_train_table)
                num_train_logged += 1
            plt.close("all")
        print(f"Epoch: {epoch}, Loss: {loss}")
        train_artifact.add(train_table, "train_predictions")
        wandb.run.log_artifact(train_artifact)
            
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
            
            valid_interm_pred, valid_pred = model(valid_inputs)
            valid_loss = criterion(valid_pred, valid_labels)
            
            # Save predictions for each epoch
            if num_logged < num_predictions_to_log:
                # print(f"Saving predictions for epoch {epoch} step {i}")
                input_img = valid_inputs.squeeze(0).squeeze(0).cpu().numpy()
                label_img = valid_labels[0][1].cpu().numpy()
                pred_img = np.argmax(valid_pred[0].detach().cpu(), 0).numpy()
                # pred_img = valid_pred[0].detach().cpu().numpy()
                
                log_predictions(input_img, label_img, pred_img, epoch, i, valid_table) # adds row to valid table
                log_predictions(input_img, label_img, pred_img, epoch, i, total_table) # adds row to total table
                num_logged += 1
                
            wandb.log({"valid_loss": valid_loss})
            plt.close("all")
        valid_artifact.add(valid_table, "predictions")
        wandb.run.log_artifact(valid_artifact)

        try:
            print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
        except:
            print(f"Epoch: {epoch} | Loss: {loss}")
        print(f"Time elapsed: {time.time() - start} seconds")
        checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss, batch_size=batch_size, lr=lr, focal_loss_weights=(criterion.gamma, criterion.alpha), path=os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
    wandb.log({"Table" : total_table})
    wandb.finish()
    print(f"Training complete. Time elapsed: {time.time() - start} seconds")

def train_with_intermediate_pred(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,lr: float,model_folder: str, model_name: str, num_predictions_to_log:int=5) -> None:
    """ 
    Train the model and log predictions to wandb.
    
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
    depth, height, width = 5, 256, 256
    total_table = wandb.Table(columns=['Epoch', 'Image'])
    total_train_table = wandb.Table(columns=['Epoch', 'train_image'])
    start = time.time()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):

        train_artifact = wandb.Artifact(f"train" + str(wandb.run.id), type="predictions")
        train_table = wandb.Table(columns=['Epoch', 'Image'])
        num_train_logged = 0
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
            intermediate_pred, pred = model(inputs)
            loss = criterion(intermediate_pred, pred, labels)
            loss.backward() # calculate gradients
            optimizer.step() # update weights based on calculated gradients
            print(f"Step: {i}, Loss: {loss}")
            wandb.log({"loss": loss})

            # Save predictions for each epoch
            if num_train_logged < num_predictions_to_log:
                print(f"Saving predictions for epoch {epoch} step {i}")
                input_img = inputs.squeeze(0).squeeze(0).cpu().numpy()
                label_img = labels[0][1].cpu().numpy()
                pred_img = np.argmax(pred[0].detach().cpu(), 0).numpy()
                # pred_img = pred[0].detach().cpu().numpy()
                
                log_predictions(input_img, label_img, pred_img, epoch, i, train_table)
                log_predictions(input_img, label_img, pred_img, epoch, i, total_train_table)
                num_train_logged += 1
            plt.close("all")
        train_artifact.add(train_table, "train_predictions")
        wandb.run.log_artifact(train_artifact)
            
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
            
            valid_intermediate_pred, valid_pred = model(valid_inputs)
            valid_loss = criterion(valid_intermediate_pred, valid_pred, valid_labels)
            print(f"Validation Step: {i}, input size: {valid_inputs.shape}, Loss: {valid_loss}")
            
            # Save predictions for each epoch
            if num_logged < num_predictions_to_log:
                print(f"Saving predictions for epoch {epoch} step {i}")
                input_img = valid_inputs.squeeze(0).squeeze(0).cpu().numpy()
                label_img = valid_labels[0][1].cpu().numpy()
                pred_img = np.argmax(valid_pred[0].detach().cpu(), 0).numpy()
                # pred_img = valid_pred[0].detach().cpu().numpy()
                
                log_predictions(input_img, label_img, pred_img, epoch, i, valid_table) # adds row to valid table
                log_predictions(input_img, label_img, pred_img, epoch, i, total_table) # adds row to total table
                num_logged += 1
                
            wandb.log({"valid_loss": valid_loss})
            plt.close("all")
        valid_artifact.add(valid_table, "predictions")
        wandb.run.log_artifact(valid_artifact)

        print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
        print(f"Time elapsed: {time.time() - start} seconds")
        checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss, batch_size=batch_size, lr=lr, focal_loss_weights=criterion.alpha, path=os.path.join(model_folder, f"{model_name}_epoch_{epoch}.pth"))
    wandb.log({"Table" : total_table})
    wandb.finish()
    print(f"Training complete. Time elapsed: {time.time() - start} seconds")


