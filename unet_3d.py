# import sys
# sys.path.append('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')

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

    from torch.utils.data import DataLoader
    from utilities import *
    from models import *
    from dataset import *
    from loss import *




    # Define directories
    data_dir = "tiniest_data"
    x_train_dir = os.path.join(data_dir, "original", "train")
    y_train_dir = os.path.join(data_dir, "ground_truth", "train")
    x_valid_dir = os.path.join(data_dir, "original", "valid")
    y_valid_dir = os.path.join(data_dir, "ground_truth", "valid")
    x_test_dir = os.path.join(data_dir, "original", "test")
    y_test_dir = os.path.join(data_dir, "ground_truth", "test")

    model_folder = "models"
    sample_preds_folder = "results"

    depth, height, width = np.load(os.path.join(x_train_dir, os.listdir(x_train_dir)[0])).shape

    class_labels = {
    0: "not gj",
    1: "gj",
    }

    # Get train and val dataset instances
    train_dataset = SliceDataset(x_train_dir, y_train_dir, image_dim = (depth, width, height), augmentation=None)
    valid_dataset = SliceDataset(x_valid_dir, y_valid_dir, image_dim = (depth, width, height))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8) # change num_workers as needed
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("Data loaders created.")
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    # Check if GPU is available
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
    print("Model initialized.")

    w1, w2 = 0.2, 0.2
    alpha = torch.Tensor([w1, w2/9])
    gamma = 3
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
    table = wandb.Table(columns=['ID', 'Image'])


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
