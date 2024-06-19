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
    "new3d": r"/home/mishaalk/scratch/gapjunc/train_datasets/final_jnc_only_split3d", 
    "test": r"/home/mishaalk/scratch/gapjunc/test_datasets/sec100_125_split"
}
                
def make_dataset_new(dataset_dir, aug=False, neuron_mask=False, mito_mask=False, chain_length=False):
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
    elif "test" in dataset_dir:
        train = DebugDataset(dataset_dir)
        valid = None
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

def train_loop(model, train_loader, criterion, optimizer, valid_loader=None, mem_feat=False, epochs=30, decay=None):
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

    recall_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(gt[gt == 1])
    precision_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(pred[pred == 1])

    for epoch in range(epochs):
        pbar = tqdm(total=len(train_loader))
        for i, data in enumerate(train_loader):
            pbar.set_description("Progress: {:.2%}".format(i/len(train_loader)))
            inputs, labels, neuron_mask, mito_mask = data # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if neuron_mask != []: neuron_mask = neuron_mask.to(DEVICE)
            if mito_mask != []: mito_mask = mito_mask.to(DEVICE)
            pred = model(inputs) if not mem_feat else model(inputs, neuron_mask.to(torch.float32))
            if args.focalweight == 0:
                loss = criterion(pred.squeeze(1), labels.squeeze(1), neuron_mask.squeeze(1) if neuron_mask != [] and not mem_feat else [], mito_mask.squeeze(1) if mito_mask != [] else [], model.s1, model.s2)
            else: loss = criterion(pred.squeeze(1), labels.squeeze(1), neuron_mask.squeeze(1) if neuron_mask != [] and not mem_feat else [], mito_mask.squeeze(1) if mito_mask != [] else [])
            loss.backward() # calculate gradients (backpropagation)
            
            # log some metrics
            wandb.log({"train_precision": precision_f(pred >= 0.5, labels)})
            wandb.log({"train_recall": recall_f(pred >= 0.5, labels)})

            if args.mask_neurons and args.gendice: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step() # update model weights (values for kernels)
            pbar.set_postfix(step = f"Step: {i}", loss = f"Loss: {loss}")
            loss_list.append(loss)
            wandb.log({"loss": loss})
            pbar.update(1)
            pbar.refresh()
            # break

        val_loss_avg, val_count = [], []
        val_prec_avg, val_recall_avg = [], []
        with torch.no_grad():
                    
            epoch_non_empty = False

            for i, data in enumerate(valid_loader):
                valid_inputs, valid_labels, valid_neuron_mask, valid_mito_mask = data
                valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
                if valid_neuron_mask != []: valid_neuron_mask = valid_neuron_mask.to(DEVICE)
                if valid_mito_mask != []: valid_mito_mask = valid_mito_mask.to(DEVICE)
                valid_pred = model(valid_inputs) if not mem_feat else model(valid_inputs, valid_neuron_mask.to(torch.float32))
                if args.focalweight == 0:
                    valid_loss = criterion(valid_pred.squeeze(1), valid_labels.squeeze(1), valid_neuron_mask.squeeze(1) if valid_neuron_mask != [] and not mem_feat else [], valid_mito_mask if valid_mito_mask !=[] else [], model.s1, model.s2)
                else:   
                    valid_loss = criterion(valid_pred.squeeze(1), valid_labels.squeeze(1), valid_neuron_mask.squeeze(1) if valid_neuron_mask != [] and not mem_feat else [], valid_mito_mask if valid_mito_mask !=[] else [])
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
                # wandb.log({"valid_loss": valid_loss})
                val_loss_avg.append(valid_loss.detach().item())
                val_count.append(valid_inputs.shape[0])
                uniques = np.unique(torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))).detach().cpu().numpy())
                if len(uniques) == 2:
                    if not epoch_non_empty:
                        epoch_non_empty = True
                        print("UNIQUE OUTPUTS!")
                else:
                    epoch_non_empty = False
                val_prec_avg.append(precision_f(valid_pred >= 0.5, valid_labels).detach().item())
                val_recall_avg.append(recall_f(valid_pred >= 0.5, valid_labels).detach().item())
            # log some metrics
            valid_loss = np.sum(np.array(val_loss_avg) * np.array(val_count))/sum(val_count)
            wandb.log({"valid_loss": val_loss_avg})
            val_prec_avg = np.sum(np.array(val_prec_avg) * np.array(val_count))/sum(val_count)
            val_recall_avg = np.sum(np.array(val_recall_avg) * np.array(val_count))/sum(val_count)
            wandb.log({"valid_precision": val_prec_avg})
            wandb.log({"valid_recall": val_recall_avg})
        if decay is not None: decay.step(valid_loss)

        print(f"Epoch: {epoch} | Loss: {loss} | Valid Loss: {valid_loss} | Valid Prec: {val_prec_avg} | Valid Recall: {val_recall_avg}")
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
    parser.add_argument("--tversky", action="store_true")
    parser.add_argument("--focalweight", default=0.5, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--mem_feat", action="store_true")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout for neural network training")
    parser.add_argument("--spatial", action="store_true", help="Spatial Pyramid")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--resnet", action="store_true")

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
            train_dataset, valid_dataset =  make_dataset_new(train_dir, aug=args.aug,neuron_mask=args.mask_neurons or args.mem_feat, mito_mask=args.mask_mito, chain_length=args.td)
        else: make_dataset_old(args.aug)

        if args.dataset is None: print("----WARNING: RUNNING OLD DATASET----")

        print("ARGS: ", sys.argv)

        #set the model and results path
        model_folder += args.dataset+f"{'_residual' if args.residual else ''}"+"/" if not args.spatial else args.dataset+f"spatial{'_residual' if args.residual else ''}"+"/"
        sample_preds_folder += args.dataset + "/" if not args.spatial else args.dataset+f"spatial{'_residual' if args.residual else ''}"+"/"

        
        if "test" not in args.dataset: 
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
            valid_loader = DataLoader(valid_dataset, batch_size=min(batch_size, 16), shuffle=False, num_workers=4)
        else:
            print(len(train_dataset))
            train_old, _ = make_dataset_new(DATASETS["new"], aug=args.aug,neuron_mask=args.mask_neurons or args.mem_feat, mito_mask=args.mask_mito, chain_length=args.td)
            print(len(train_old))
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [math.ceil(0.08*len(train_dataset)), math.ceil(0.12*len(train_dataset)), len(train_dataset) - math.ceil(0.08*len(train_dataset)) - math.ceil(0.12*len(train_dataset))])
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_old]) 
            print(len(train_dataset))          
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
        print("Data loaders created.")

        DEVICE = torch.device("cuda") if torch.cuda.is_available() or not args.cpu else torch.device("cpu")

        if args.model_name is None and not args.resnet:
            model = UNet(three=args.td, scale=args.focalweight == 0, spatial=args.spatial, dropout=args.dropout, residual=args.residual).to(DEVICE) if not args.mem_feat else MemUNet().to(DEVICE)
        elif args.model_name is None and args.resnet:
            model = ResUNet(three=args.td).to(DEVICE) if not args.mem_feat else MemResUNet(three=args.td).to(DEVICE)
        else: model = joblib.load(os.path.join("/home/mishaalk/scratch/gapjunc/models/", args.model_name)) # load model

        if not args.infer:

            print("Current dataset {}".format(args.dataset))

            if args.dataset != "tiny" and args.dataset != "erased" and args.dataset != "new" and args.dataset != "new3d" and args.dataset != "test":
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
                cls_weights = torch.Tensor([0.05, 0.95])

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
                floss, gloss =  FocalLoss(alpha=cls_weights, device=DEVICE), GenDLoss()
                if args.focalweight == 0:
                    criterion = lambda a,b,c,d,s1, s2: floss(s1*a, b, c, d) + gloss(s2*a, b, c, d)
                else: criterion = lambda a,b,c,d: args.focalweight * floss(a, b, c, d) + (1 - args.focalweight) *gloss(a, b, c, d)
                model_folder =model_folder[:-1] +"dicefocalloss"
            elif args.tversky:
                criterion = TverskyFocal()
                model_folder =model_folder[:-1] +"tvrsky"
            else:
                criterion = FocalLoss(alpha=cls_weights, device=DEVICE)#torch.nn.BCEWithLogitsLoss()
                model_folder= model_folder[:-1] +"focal"

            optimizer = torch.optim.Adam(model.parameters(),lr=1e-6)
            decayed_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            loss_list = [] 

            if args.split:
                classifier_criterion = torch.nn.BCEWithLogitsLoss()
            

            setup_wandb(30, 0.0001)

            print("Starting training...")
            start = time.time()

            #determine save folder
            counter = 0
            model_folder+="mito" if args.mask_mito else ""
            model_folder += "neuro" if args.mask_neurons else ""
            while (os.path.isdir(model_folder)):
                model_folder += str(counter)
                counter+=1
                continue
            os.mkdir(model_folder)

            print("SAVING MODELS TO {}".format(model_folder))
    
            if not args.split:
                print(f"running for {300 if not args.epochs else args.epochs} epochs")
                train_loop(model, train_loader, criterion, optimizer, valid_loader, epochs=300 if not args.epochs else args.epochs, mem_feat=args.mem_feat)
            else:
                print(f"running for {200 if not args.epochs else args.epochs} epochs")
                train_loop_split(model, train_loader, classifier_criterion, criterion, optimizer, valid_loader, epochs=200 if not args.epochs else args.epochs, mem_feat=args.mask_feat)
                
        else:

            if not args.split:
                inference_save(model, train_dataset, valid_dataset)
            else:
                inference_save_split(model, train_dataset, valid_dataset)

    else:
        #--- Personal debug area ---

        model_folder += "2d_resnet_focal_run1"
        sample_preds_folder = sample_preds_folder+"/2d_resnet_focal_run1_test"
        model = joblib.load(os.path.join(model_folder, "model5_epoch126.pk1")) #3d gendice epoch 95, focal epoch 108, 2d mask 115, df_0.2 R1 73, dyna R1 82
        # for flat dyna we have 125, 2wt we have 142, 2d_gd_mem_run2 best was 36, 2d_membrane_aug_low is 83, 2d_membrane_noaug is 140
        #resnet w memebrane 67 or 84, 
        #resnet 3d 131 or 130
        #resnet focal 126
        model = model.to("cuda")
        model.eval()

        # dataset_dir = "/home/mishaalk/scratch/gapjunc/train_datasets/final_jnc_only_split/"
        # dataset_dir = "/home/mishaalk/scratch/gapjunc/test_datasets/3d_test_new/imgs/"
        dataset_dir = "/home/mishaalk/scratch/gapjunc/test_datasets/sec100_125_split/imgs/"

        # dataset = CaImagesDataset(dataset_dir, preprocessing=None, augmentation=None, image_dim=(512, 512), split=1)
        dataset = TestDataset(dataset_dir, td=False, membrane="/home/mishaalk/scratch/gapjunc/test_datasets/sec100_125_split/neurons/")
        imgs_files, gt_files = [i for i in sorted(os.listdir(dataset_dir)) if "DS" not in i], [i for i in sorted(os.listdir(dataset_dir)) if "DS" not in i]
        def collate_fn(batch):
            return (
                torch.stack([x[0] for x in batch]),
                torch.stack([x[1] for x in batch]),
                torch.stack([x[2] for x in batch]),
                torch.stack([x[3] for x in batch]),
            )
        dataloader = DataLoader(dataset, batch_size=36, shuffle=False, num_workers=8)

        if not os.path.isdir(sample_preds_folder):
            os.mkdir(sample_preds_folder)

        # assert len(imgs_files) == 2695
        recall_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(gt[gt == 1])
        precision_f = lambda pred, gt: torch.nansum(pred[(pred == 1) & (gt == 1)])/torch.nansum(pred[pred == 1])

        i = 0
        with torch.no_grad():
            class_list = []
            rec_acc, prec_acc = [], []
            for data in tqdm(dataloader):
                image, gt, membrane, _ = data

                #infer:
                x_tensor = image.to("cuda")
                memb = membrane.to("cuda")
                pred_mask = model(x_tensor) # [1, 2, 512, 512]
                # print(pred_mask.shape)
                # pred_mask_binary = pred_mask.squeeze(0).detach()
                pred_mask_binary = pred_mask
                # classified = torch.round(nn.Sigmoid()(classified))
                pred_mask_binary = pred_mask_binary.to("cpu")
                for j in range(pred_mask_binary.shape[0]):
                    file_name = imgs_files[i]
                    # prfed_og = cv2.cvtColor(cv2.imread(sample_preds_folder+file_name), cv2.COLOR_BGR2GRAY)
                    # print(np.count_nonzero(prfed_og == pred_mask_binary[j].squeeze(0).numpy()))
                    # raise Exception
                    # if os.path.isfile(sample_preds_folder+file_name): continue
                    # cv2.imwrite(sample_preds_folder+file_name+".png", pred_mask_binary[j].squeeze(0).numpy())
                    #save it as a numpy file
                    f = open(sample_preds_folder+file_name+".npy", "wb")
                    np.save(f, pred_mask_binary[j].squeeze(0).numpy())
                    f.close()
                    i+=1
                    # class_list.append([file_name, classified[j].item()])
                    pred_mask_binary1 = nn.Sigmoid()(pred_mask_binary) >= 0.5
                    gt[gt == 255] = 1
                    rec_acc.append(recall_f(pred_mask_binary1, gt).detach().item())
                    prec_acc.append(precision_f(pred_mask_binary1, gt).detach().item())
        print(np.nanmean(prec_acc))
        print(np.nanmean(rec_acc))



        print(f"I wrote {i} files")
        
        # with open(sample_preds_folder+"classes.csv", "w") as f:
        #     obj = csv.writer(f)
        #     obj.writerows(class_list)
            

