from pathlib import Path
import torch

from torchvision.transforms import v2
import wandb
import os
import time
import cv2
import joblib
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm 

import dataset
import setmodels

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x_train_dir=r"E:\Mishaal\GapJunction\small_data\original\train"
y_train_dir=r"E:\Mishaal\GapJunction\small_data\ground_truth\train"

x_valid_dir=r"E:\Mishaal\GapJunction\small_data\original\valid"
y_valid_dir=r"E:\Mishaal\GapJunction\small_data\ground_truth\valid"

x_test_dir=r"E:\Mishaal\GapJunction\small_data\original\test"
y_test_dir=r"E:\Mishaal\GapJunction\small_data\ground_truth\test"

model_folder = r"E:\Mishaal\GapJunction\models"
sample_preds_folder = r"E:\Mishaal\GapJunction\results"

height, width = cv2.imread(os.path.join(x_train_dir, os.listdir(x_train_dir)[0])).shape[:2]

WANDB_API_KEY = "42a2147c44b602654473783bde1ecd15579cc313"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

#suitable good transforms for this:
"""
Random Horizontal Flip
Random Vrtitcal Flip
Random Rotation
Random Affine
Random Perspective
Gaussian Blur
"""
def train(model, train_loader, val_loader, lr=1e-3, epochs=30, model_name="modelX", augments = False):
    global model_folder

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
    0: "background",
    1: "gapjunction",
    }

    print("Starting training...")
    start = time.time()
    alpha = 0.25  
    gamma = 3
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_list = [] 
    table = wandb.Table(columns=['ID', 'Image'])

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            print("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, labels = data # (inputs: [batch_size, 3, 512, 512], labels: [batch_size, 3, 512, 512])
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            pred = model(inputs)
            bce_loss = criterion(pred, labels) # calculate loss (binary cross entropy)
            p_t = torch.exp(-bce_loss)
            focal_loss = alpha* (1 - p_t) ** gamma * bce_loss
            loss = focal_loss.mean()
            loss.backward() # calculate gradients (backpropagation)
            optimizer.step() # update model weights (values for kernels)
            print(f"Step: {i}, Loss: {loss}")
            loss_list.append(loss)
            wandb.log({"loss": loss})

        for i, data in enumerate(val_loader):
            valid_inputs, valid_labels = data
            valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
            valid_pred = model(valid_inputs)
            bce_loss = criterion(valid_pred, valid_labels) # calculate loss (binary cross entropy)
            p_t = torch.exp(-bce_loss)
            focal_loss = alpha* (1 - p_t) ** gamma * bce_loss
            valid_loss = focal_loss.mean()
            mask_img = wandb.Image(valid_inputs[0].squeeze(0).cpu().numpy(), 
                                    masks = {
                                        "predictions" : {
                            "mask_data" : (valid_pred[0] * 255 ).detach().cpu().numpy(),
                            "class_labels" : class_labels
                        },
                        "ground_truth" : {
                            "mask_data" : (valid_labels[0] * 255).cpu().numpy(),
                            "class_labels" : class_labels
                        }}
            )
            table.add_data(f"Epoch {epoch} Step {i}", mask_img)
            wandb.log({"valid_loss": valid_loss})

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

def generate_masks(dataset, model_name, split="train"):
    sample_train_folder = sample_preds_folder+ ("\\train_res" if split == "train" else "\\val_res")
    model = joblib.load(os.path.join(model_folder, model_name))
    model = model.to(DEVICE)
    model.eval()
    for i in tqdm(range(len(dataset))):
        image, gt_mask = dataset[i] # image and ground truth from test dataset
        # print(image.shape, gt_mask.shape) # [1, 512, 512] and [2, 512, 512]
        # print(image)
        suffix = "_1_{}".format(i)
        plt.imshow(image.squeeze(0).numpy(), cmap='gray')
        plt.savefig(os.path.join(sample_train_folder, f"sample_pred_{suffix}.png"))
        # plt.show()
        plt.imshow(gt_mask[1].numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_train_folder, f"sample_gt_{suffix}.png"))
        # plt.show()
        x_tensor = image.to(DEVICE).unsqueeze(0)
        pred_mask = model(x_tensor) # [1, 2, 512, 512]
        # print(pred_mask.shape)
        # pred_mask_binary = pred_mask.squeeze(0).detach()
        pred_mask_binary = torch.round(torch.nn.Sigmoid()(pred_mask)) * 255
        plt.imshow(pred_mask_binary.cpu().numpy(), cmap="gray")
        plt.savefig(os.path.join(sample_train_folder, f"sample_pred_binary_{suffix}.png"))

if __name__ == "__main__":
    # Get train and val dataset instances
    train_dataset = dataset.SectionsDataset(
        x_train_dir, y_train_dir, 
        augmentation=None,
        preprocessing=None,
        image_dim = (width, height)
    )
    valid_dataset = dataset.SectionsDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=None,
        preprocessing=None,
        image_dim = (width, height)
    )

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("Data loaders created.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--modelname", type=str, default="")
    parser.add_argument("--augment", action="store_true")

    args = parser.parse_args()

    if args.train:
        train(setmodels.SetNet() if args.modelname == "" else joblib.load(os.path.join(model_folder, args.modelname)), train_loader, valid_loader, augments=args.augment)
    if args.val:
        generate_masks(train_dataset, model_name=args.modelname)
        generate_masks(valid_dataset, model_name=args.modelname, split="val")
