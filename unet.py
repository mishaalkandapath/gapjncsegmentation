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
import sys, csv

model_folder = r"/home/mishaalk/scratch/gapjunc/models/"
sample_preds_folder = r"/home/mishaalk/scratch/gapjunc/results/"
table, class_labels = None, None #wandb stuff

torch.autograd.set_detect_anomaly(True)
DATASETS = {
    "new": r"/home/mishaalk/scratch/gapjunc/train_datasets/final_jnc_only_split", 
    "tiny": r"/home/mishaalk/scratch/gapjunc/train_datasets/final_tiny_jnc_only128",
    "new3d": r"/home/mishaalk/scratch/gapjunc/train_datasets/final_jnc_only_split3d"
}
                
def make_dataset_new(x_new_dir, y_new_dir, aug=False, neuron_mask=False, mito_mask=False, chain_length=False):
    try:
        height, width = cv2.imread(os.path.join(x_new_dir, os.listdir(x_new_dir)[0])).shape[:2]
    except:
        height, width = 512, 512

    # Get train and val dataset instances
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.4)
    ])
    if chain_length:
        train = SectionsDataset(
            dataset_dir,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            chain_length=chain_length,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask
        )
        valid = SectionsDataset(
            dataset_dir,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            chain_length=chain_length,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask, 
            split=1
        )
    else:
        train = CaImagesDataset(
            dataset_dir,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask
        )
        valid = CaImagesDataset(
            dataset_dir,
            preprocessing=None,
            image_dim = (width, height), augmentation=augmentation if aug else None,
            mask_neurons= neuron_mask,
            mask_mito = mito_mask, 
            split=1
        )

    return train, valid

def train_loop(model, train_loader, criterion, optimizer, valid_loader=None, epochs=30, decay=None):
    global table, class_labels, model_folder, DEVICE, args
    
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
    signal.signal(signal.SIGUSR1, sigint_handler)

    for epoch in range(epochs):
        pbar = tqdm(total=len(train_loader))
        for i, data in enumerate(train_loader):
            pbar.set_description("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, labels, neuron_mask, mito_mask = data # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if neuron_mask != []: neuron_mask = neuron_mask.to(DEVICE)
            if mito_mask != []: mito_mask = mito_mask.to(DEVICE)
            pred = model(inputs)
            loss = criterion(pred.squeeze(1), labels.squeeze(1), neuron_mask.squeeze(1) if neuron_mask != [] else [], mito_mask.squeeze(1) if mito_mask != [] else [])
            loss.backward() # calculate gradients (backpropagation)
            # for name, param in model.named_parameters():
            #     print(name, torch.any(torch.isnan(param.grad)))
            # raise Exception
            if args.mask_neurons and args.gendice: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step() # update model weights (values for kernels)
            pbar.set_postfix(step = f"Step: {i}", loss = f"Loss: {loss}")
            loss_list.append(loss)
            wandb.log({"loss": loss})
            pbar.update(1)
            pbar.refresh()
            # break

        with torch.no_grad():
                    
            epoch_non_empty = False

            for i, data in enumerate(valid_loader):
                valid_inputs, valid_labels, valid_neuron_mask, valid_mito_mask = data
                valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
                if valid_neuron_mask != []: valid_neuron_mask = valid_neuron_mask.to(DEVICE)
                if valid_mito_mask != []: valid_mito_mask = valid_mito_mask.to(DEVICE)
                valid_pred = model(valid_inputs)
                valid_loss = criterion(valid_pred.squeeze(1), valid_labels.squeeze(1), valid_neuron_mask.squeeze(1) if valid_neuron_mask != [] else [], valid_mito_mask if valid_mito_mask !=[] else [])
                mask_img = wandb.Image(valid_inputs[0].squeeze(0).cpu().numpy()[0] if args.td else valid_inputs[0].squeeze(0).cpu().numpy(), 
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

    sample_train_folder = sample_preds_folder+"\\\\train_res"
    model = joblib.load(os.path.join(model_folder, "model5_epoch89.pk1"))
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

    sample_val_folder = sample_preds_folder+"\\\\valid_res"
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


def train_loop_split(model, train_loader, classifier_criterion, criterion, optimizer, valid_loader=None, epochs=30, decay=None):
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
    signal.signal(signal.SIGUSR1, sigint_handler)
    
    for epoch in range(epochs):
        pbar = tqdm(total = len(train_loader))
        for i, data in enumerate(train_loader):
            pbar.set_description("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, labels, neuron_mask, mito_mask= data # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if neuron_mask != []: neuron_mask = neuron_mask.to(DEVICE)
            if mito_mask != []: mito_mask = mito_mask.to(DEVICE)
            train_class_labels = labels.view(labels.shape[0], -1).sum(axis=-1, keepdim=True) >= 1 # loss mask

            pred, classifier = model(inputs)
            classifier_loss = classifier_criterion(classifier, train_class_labels.to(dtype=torch.float32)).mean()

            #might have to scale the criteria, decide from experiments
            if neuron_mask == []:
                loss = criterion(pred, labels, train_class_labels, mito_mask) * batch_size/torch.count_nonzero(train_class_labels) # take into account only the losses that have positive labels
            else:
                neuron_mask *= train_class_labels.unsqueeze(-1)
                mito_mask *= train_class_labels.unsqueeze(-1)
                loss = criterion(pred, labels, neuron_mask, mito_mask) * batch_size/torch.count_nonzero(train_class_labels) # take into account only the losses that have positive labels

            wandb.log({"segmentation loss": loss})
            loss += classifier_loss
            loss.backward() # calculate gradients (backpropagation)
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step() # update model weights (values for kernels)
            pbar.set_postfix(step=f"Step: {i}", loss=f"Loss: {loss}")
            loss_list.append(loss)
            wandb.log({"loss": loss})
            wandb.log({"classifier loss": classifier_loss})
            pbar.update(1)
            pbar.refresh()
        
        epoch_non_empty = False

        for i, data in enumerate(valid_loader):
            valid_inputs, valid_labels, v_neuron_mask, v_mito_mask = data
            valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
            if v_neuron_mask != []: v_neuron_mask = v_neuron_mask.to(DEVICE)
            if v_mito_mask != []: v_mito_mask = v_mito_mask.to(DEVICE)
            valid_class_labels = valid_labels.view(valid_labels.shape[0], -1).sum(axis=-1, keepdim=True) >= 1 # loss mask

            valid_pred, valid_classifier = model(valid_inputs)
            valid_classifier_loss = classifier_criterion(valid_classifier, valid_class_labels.to(dtype=torch.float32)).mean()

            if v_neuron_mask == []:
                valid_loss = criterion(valid_pred, valid_labels, valid_class_labels, v_mito_mask)
            else:
                v_neuron_mask *= valid_class_labels.unsqueeze(-1)
                v_mito_mask *= valid_class_labels.unsqueeze(-1)
                valid_loss = criterion(valid_pred, valid_labels, v_neuron_mask, v_mito_mask)
            valid_loss += valid_classifier_loss
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
            wandb.log({"classification loss": valid_classifier_loss})
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

def inference_save_split(model, train_dataset, valid_dataset):
    global DEVICE, model_folder, sample_preds_folder

    sample_train_folder = sample_preds_folder+"//train_res"
    model = joblib.load(os.path.join(model_folder, "model5_epoch17.pk1"))
    model = model.to(DEVICE)
    model.eval()
    for i in tqdm(range(len(train_dataset))):
        image, gt_mask, _ = train_dataset[i] # image and ground truth from test dataset
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
        image, gt_mask, _ = valid_dataset[i] # image and ground truth from test dataset
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
        dir="/home/mishaalk/scratch/gapjunc",
        
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

    table = wandb.Table(columns=['ID', 'Image'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mask_neurons", action="store_true")
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--mask_mito", action="store_true")
    parser.add_argument("--special", action="store_true")
    parser.add_argument("--td", action="store_true")
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--dice", action="store_true")
    parser.add_argument("--gendice", action="store_true")
    parser.add_argument("--ss", action="store_true")
    parser.add_argument("--dicefocal", action="store_true")
    parser.add_argument("--focalweight", default=0.5, type=int)

    args = parser.parse_args()

    if not args.special:

        if args.seed:
            SEED = 12
            np.random.seed(SEED)
            random.seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.manual_seed(SEED)
            os.environ["PYTHONHASHSEED"] = str(SEED)

        batch_size = args.batch_size

        
        if args.dataset is not None:
            train_dir = (DATASETS[args.dataset])
            if args.td:
                train_dataset, valid_dataset =  make_dataset_3d(train_dir aug=args.aug,neuron_mask=args.mask_neurons, mito_mask=args.mask_mito)
            else:
                train_dataset, valid_dataset =  make_dataset_new(train_dir, aug=args.aug,neuron_mask=args.mask_neurons, mito_mask=args.mask_mito)
        else: make_dataset_old(args.aug)



        if args.dataset is None: print("----WARNING: RUNNING OLD DATASET----")

        #set the model and results path
        model_folder += args.dataset+"/"
        sample_preds_folder += args.dataset + "/" 

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=min(batch_size, 16), shuffle=False, num_workers=4)
        print("Data loaders created.")

        DEVICE = torch.device("cuda") if torch.cuda.is_available() or not args.cpu else torch.device("cpu")

        if args.model_name is None: model = UNet(three=args.td).to(DEVICE) if not args.split else SplitUNet(three=args.td).to(DEVICE)
        else: model = joblib.load(os.path.join("/home/mishaalk/scratch/gapjunc/models/", args.model_name)) # load model

        if not args.infer:

            print("Current dataset {}".format(args.dataset))

            if args.dataset != "tiny" and args.dataset != "erased" and args.dataset != "new" and args.dataset != "new3d":
                print("MASKING NEURONS IS SET TO {}".format(args.mask_neurons))

                #calc focal weighting:
                smushed_labels = None
                if args.mask_neurons: nm = None
                for i in tqdm(range(len(train_dataset))):
                    if smushed_labels is None: smushed_labels = train_dataset[i][1].to(torch.int64)
                    else: smushed_labels = torch.concat([smushed_labels, train_dataset[i][1].to(torch.int64)])
                    if args.mask_neurons and nm is None: nm = torch.tensor(train_dataset[i][2], dtype=torch.bool)
                    elif args.mask_neurons: nm = torch.concat([nm,  torch.tensor(train_dataset[i][2])], dtype=torch.bool)

                smushed_labels = smushed_labels.flatten() if not args.mask_neurons else smushed_labels.flatten()[nm.flatten()]
                class_counts = torch.bincount(smushed_labels)
                total_samples = len(train_dataset) * 512 * 512
                w1, w2 = 1/(class_counts[0]/total_samples), 1/(class_counts[1]/total_samples)
                cls_weights = torch.Tensor([w1, w2/2]) #soften it a bit
                print(cls_weights)
            elif args.dataset == "tiny":
                cls_weights = torch.Tensor([1, 90])
                print(cls_weights)
            elif args.dataset == "erased":
                cls_weights = torch.Tensor([1, 15])
            else:
                cls_weights = torch.Tensor([1, 24])

            #init oprtimizers
            if args.gendice:
                criterion = GenDLoss()
                model_folder= model_folder[:-1] + "gendice"
            elif args.dice:
                criterion = DiceLoss()
                model_folder = model_folder[:-1] + "dice"
            elif args.ss:
                criterion = SSLoss()
                model_folder =model_folder[:-1] +"ssloss"
            elif args.dicefocal:
                floss, GenDLoss =  FocalLoss(alpha=cls_weights, device=DEVICE), GenDLoss()
                criterion = lambda a,b,c,d: args.focalweight * floss(a, b, c, d) + (1 - args.focalweight) *gloss(a, b, c, d)
                model_folder =model_folder[:-1] +"dicefocalloss"
            else:
                criterion = FocalLoss(alpha=cls_weights, device=DEVICE)#torch.nn.BCEWithLogitsLoss()
                model_folder= model_folder[:-1] +"focal"

            optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
            decayed_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            loss_list = [] 

            if args.split:
                classifier_criterion = torch.nn.BCEWithLogitsLoss()
            

            setup_wandb(30, 0.0001)

            print("Starting training...")
            start = time.time()

            print("SAVING MODELS TO {}".format(model_folder))

            if not args.split:
                train_loop(model, train_loader, criterion, optimizer, valid_loader, epochs=90)
            else:
                train_loop_split(model, train_loader, classifier_criterion, criterion, optimizer, valid_loader, epochs=200)
                
        else:

            if not args.split:
                inference_save(model, train_dataset, valid_dataset)
            else:
                inference_save_split(model, train_dataset, valid_dataset)

    else:
        #--- Personal debug area ---

        model_folder += "tinymitobest"
        sample_preds_folder = sample_preds_folder+"/tinymitobesttest/"
        model = joblib.load(os.path.join(model_folder, "model5_epoch114.pk1"))
        model = model.to("cuda")
        model.eval()

        # x_dir, y_dir = "/home/mishaalk/scratch/gapjunc/train_datasets/jnc_only_dataset/imgs", "/home/mishaalk/scratch/gapjunc/train_datasets/jnc_only_dataset/gts"
        x_dir, y_dir = "/home/mishaalk/scratch/gapjunc/test_datasets/tiny_test", "/home/mishaalk/scratch/gapjunc/test_datasets/tiny_test"

        dataset = CaImagesDataset(x_dir, y_dir, preprocessing=None, augmentation=None, image_dim=(512, 512))
        imgs_files, gt_files = sorted(os.listdir(x_dir)), sorted(os.listdir(y_dir))
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8)

        # assert len(imgs_files) == 2695

        i = 0
        with torch.no_grad():
            class_list = []
            for data in tqdm(dataloader):
                image, gt, _, _ = data

                #infer:
                x_tensor = image.to("cuda")
                pred_mask, _ = model(x_tensor) # [1, 2, 512, 512]
                # print(pred_mask.shape)
                # pred_mask_binary = pred_mask.squeeze(0).detach()
                pred_mask_binary = torch.round(nn.Sigmoid()(pred_mask)) * 255
                # classified = torch.round(nn.Sigmoid()(classified))
                pred_mask_binary = pred_mask_binary.to("cpu")
                for j in range(pred_mask_binary.shape[0]):
                    file_name = imgs_files[i]
                    # prfed_og = cv2.cvtColor(cv2.imread(sample_preds_folder+file_name), cv2.COLOR_BGR2GRAY)
                    # print(np.count_nonzero(prfed_og == pred_mask_binary[j].squeeze(0).numpy()))
                    # raise Exception
                    # if os.path.isfile(sample_preds_folder+file_name): continue
                    cv2.imwrite(sample_preds_folder+file_name+".png", pred_mask_binary[j].squeeze(0).numpy())
                    i+=1
                    # class_list.append([file_name, classified[j].item()])

        print(f"I wrote {i} files")
        
        # with open(sample_preds_folder+"classes.csv", "w") as f:
        #     obj = csv.writer(f)
        #     obj.writerows(class_list)
            

