# import packages
import os
import numpy as np;
import torch 

from utilities import *
from models import *
from dataset import *
from loss import *
from utilities_train import *

if __name__ == "__main__":  
    # --- Setup device, random seed, and parse arguments ---
    # Setup device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Parse arguments
    args = parse_arguments()
   
    # Setup random seed
    seed = args.seed
    print("seed:", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    # ----- Define class labels and directories -----
    # Define class labels
    class_labels = {
    0: "not gj", 
    1: "gj",
    }
    
    # Define model directory
    model_name = args.model_name
    model_folder = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_folder): os.makedirs(model_folder)
    
    # Define data directory
    data_dir = args.data_dir
    if not os.path.exists(data_dir): print(f"Data directory {data_dir} does not exist.")
    
    # Define results directory
    results_folder = os.path.join(args.results_dir, args.model_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        os.makedirs(os.path.join(results_folder, "train"))
        os.makedirs(os.path.join(results_folder, "valid"))
    # ----- Load data -----
    batch_size = args.batch_size
    num_workers = args.num_workers
    print("Augment", args.augment)
    if args.cellmask_dir is not None:
        print("Using cell mask -- 2 class prediction")
        train_dataset, valid_dataset, train_loader, valid_loader = setup_datasets_and_dataloaders_withmemb(data_dir, args.cellmask_dir, batch_size, num_workers, augment=args.augment)
    else:
        train_dataset, valid_dataset, train_loader, valid_loader = setup_datasets_and_dataloaders(data_dir, batch_size, num_workers, augment=args.augment)
    print(f"Batch size: {batch_size}, Number of workers: {num_workers}")
    print(f"Data loaders created. Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    # ----- Initialize model, loss function, and wandb -----
    # Initialize model
    lr = args.lr
    epochs = args.epochs
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    load_model_path = args.load_model_path
    model = model.to(DEVICE)
    if load_model_path is not None:
        model, optimizer, start_epoch, loss, batch_size, lr, focal_loss_weights = load_checkpoint(model, optimizer, load_model_path)
        print(f"Model loaded from {load_model_path}. Starting from epoch {start_epoch}.")
    print(f"Model is on device {next(model.parameters()).device}")

    # Initialize loss function
    print("Args loss type", args.loss_type)
    if args.loss_type == "combo":
        criterion = ComboLoss(alpha=args.alpha, ce_ratio=args.ce_ratio)
        print("using combo loss", "alpha:", args.alpha, "ce_ratio", args.ce_ratio)
    elif args.loss_type == "focalt":
        criterion = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        print("using focal tverskys loss", "alpha:", args.alpha, "beta:", args.beta, "gamma:", args.gamma)
    elif args.loss_type == "dicebce":
        criterion = DiceBCELoss()
        print("using dicebce loss")
    elif args.loss_type == "focal":
        alpha = torch.Tensor([args.alpha, 1-args.alpha]).to(DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=args.gamma, device=DEVICE)
        print(f"using focal loss", "alpha:", args.alpha, "gamma:", args.gamma)
    else:
        criterion = DiceLoss()
        print("using dice loss")
    print("Loss function initialized.")

    # ----- Train model -----
    train_log_local(model=model, 
          train_loader=train_loader, 
          valid_loader=valid_loader, 
          criterion=criterion, 
          optimizer=optimizer, 
          epochs=epochs, 
          batch_size=args.batch_size, 
          lr=args.lr, 
          model_folder=model_folder, 
          model_name=model_name, 
          results_folder=results_folder,
          num_predictions_to_log=args.num_predictions_to_log)
