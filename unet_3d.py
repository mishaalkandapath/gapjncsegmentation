# run if is main file
if __name__ == "__main__":
    # import packages
    import os
    import cv2
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
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a 3D U-Net model on the tiniest dataset")
    parser.add_argument("--data_dir", type=str, default="tiniest_data", help="Directory containing the tiniest dataset")
    parser.add_argument("--model_name", type=str, default="model1", help="Name of the model to save")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--w1", type=float, default=0.2, help="Weight for class 0 in Focal Loss")
    parser.add_argument("--w2", type=float, default=0.2, help="Weight for class 1 in Focal Loss")
    parser.add_argument("--gamma", type=float, default=3, help="Gamma parameter for Focal Loss")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    model_name = args.model_name
    model_folder = os.path.join("models", model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    sample_preds_folder = "results"
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Define directories
    x_train_dir = os.path.join(data_dir, "original", "train")
    y_train_dir = os.path.join(data_dir, "ground_truth", "train")
    x_valid_dir = os.path.join(data_dir, "original", "valid")
    y_valid_dir = os.path.join(data_dir, "ground_truth", "valid")
    x_test_dir = os.path.join(data_dir, "original", "test")
    y_test_dir = os.path.join(data_dir, "ground_truth", "test")

    depth, height, width = np.load(os.path.join(x_train_dir, os.listdir(x_train_dir)[0])).shape

    class_labels = {
    0: "not gj",
    1: "gj",
    }

    # Get train and val dataset instances
    train_dataset = SliceDataset(x_train_dir, y_train_dir, image_dim = (depth, width, height), augmentation=None)
    valid_dataset = SliceDataset(x_valid_dir, y_valid_dir, image_dim = (depth, width, height))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers) # change num_workers as needed
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("Data loaders created.")
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    # Check if GPU is available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Initialize model
    lr = args.lr
    epochs = args.epochs
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    load_model_path = args.load_model_path
    if load_model_path is not None:
        model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, load_model_path)
    model.train()
    print("Model initialized.")

    w1, w2 = args.w1, args.w2
    alpha = torch.Tensor([w1, w2/9])
    gamma = args.gamma
    criterion = FocalLoss(alpha=alpha, gamma=gamma, device=DEVICE)
    print("Loss function initialized.")

    wandb.init(
        project="gap-junction",
        config={
        "learning_rate": lr,
        "epochs": epochs,
        "alpha": alpha,
        }
    )      
    table = wandb.Table(columns=['Epoch', 'Image'])

    # Train model
    print("Starting training...")
    start = time.time()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            print("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.unsqueeze(1) # add channel dimension
            print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            optimizer.zero_grad() # zero gradients (otherwise they accumulate)
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward() # calculate gradients
            optimizer.step() # update weights based on calculated gradients
            print(f"Step: {i}, Loss: {loss}")
            wandb.log({"loss": loss})

        for i, data in enumerate(valid_loader):
            valid_inputs, valid_labels = data
            valid_pred = model(valid_inputs) # (N, C=2, D, H, W)
            valid_loss = criterion(valid_pred, valid_labels)
            
            # Save sample predictions as image to wandb
            # -- remove batch dim and take argmax to reverse one hot encoding -> (D, H, W)
            input_img = valid_inputs.squeeze(0).numpy()
            label_img = valid_labels[0][1].numpy()
            pred_img = np.argmax(valid_pred[0].detach(), 0).numpy()
            # -- plot as 3 rows: input, ground truth, prediction
            fig, ax = plt.subplots(3, depth, figsize=(15, 5))
            for i in range(depth):
                ax[0, i].imshow(input_img[i], cmap="gray")
                ax[1, i].imshow(label_img[i], cmap="gray")
                ax[2, i].imshow(pred_img[i], cmap="gray")
            ax[0, 0].set_ylabel("Input")
            ax[1, 0].set_ylabel("Ground Truth")
            ax[2, 0].set_ylabel("Prediction")
            mask_img = wandb.Image(fig)          
            # mask_img = wandb.Image(
            #     valid_inputs[0].squeeze(0).numpy(), # original image
            #     masks = {
            #         "predictions" : {"mask_data" : np.argmax(valid_pred[0].detach(), 0).numpy(),  
            #                          "class_labels" : class_labels},
            #         "ground_truth" : {"mask_data" : valid_labels[0][1].numpy(),"class_labels" : class_labels}
            #     }
            #     )
            table.add_data(f"Epoch {epoch} Step {i}", mask_img)
            wandb.log({"valid_loss": valid_loss})

        print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss}")
        print(f"Time elapsed: {time.time() - start} seconds")
        checkpoint(model, optimizer, epoch, loss, os.path.join(model_folder, f"{model_name}_epoch{epoch}.pk1"))
    print(f"Total time: {time.time() - start} seconds")
    wandb.log({"Table" : table})
    checkpoint(model, optimizer, epoch, loss, os.path.join(model_folder, "model1.pk1"))
    wandb.finish()
