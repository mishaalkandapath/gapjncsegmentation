# import packages
import os
import numpy as np
import torch
from utilities.utilities import *
from utilities.models import *
from utilities.dataset import *
from utilities.loss import *
from utilities.utilities_train import *

if __name__ == "__main__":
    # ----- Parse arguments -----
    args = parse_arguments()
    
    # print each of the arguments
    print("------------------------------Arguments------------------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
   
    # Setup random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define model directory
    model_name = args.model_name
    model_folder = os.path.join(args.model_dir, args.model_name)
    if not os.path.exists(model_folder): os.makedirs(model_folder)
    
    # Define results directory
    results_folder = os.path.join(args.results_dir, args.model_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        os.makedirs(os.path.join(results_folder, "train"))
        os.makedirs(os.path.join(results_folder, "valid"))
        
    loss_folder = args.loss_dir
    if not os.path.exists(loss_folder):
        os.makedirs(loss_folder)
    
    # ----- Load data -----
    print("------------------------------Loading Data------------------------------")
    batch_size = args.batch_size
    num_workers = args.num_workers
    augment = args.augment
    colour_augment = args.colour_augment
    downsample_factor = args.downsample_factor
    pred3classes = args.pred3classes
    train_x_dirs = args.train_x_dirs
    train_y_dirs = args.train_y_dirs
    valid_x_dirs = args.valid_x_dirs
    valid_y_dirs = args.valid_y_dirs
    train_cellmask_dir = args.train_cellmask_dir
    valid_cellmask_dir = args.valid_cellmask_dir
    
    # if multiple directories are provided, use them to create the dataset
    print("Setting up from lists")
    train_dataset, train_loader = setup_datasets_and_dataloaders_from_lists2d(img_dir_list=train_x_dirs, mask_dir_list=train_y_dirs, batch_size=batch_size, num_workers=num_workers, augment=augment, shuffle=True, downsample_factor=downsample_factor, colour_augment=colour_augment)
    valid_dataset, valid_loader = setup_datasets_and_dataloaders_from_lists2d(img_dir_list=valid_x_dirs, mask_dir_list=valid_y_dirs, batch_size=1, num_workers=num_workers, augment=False, shuffle=False, downsample_factor=downsample_factor, colour_augment=False)
    print(f"Batch size: {batch_size}, Number of workers: {num_workers}")
    print(f"Data loaders created. Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    # ----- Initialize model & loss function-----
    print("------------------------------Initializing Model------------------------------")
    # Initialize model
    lr = args.lr
    epochs = args.epochs
    model = UNet2D()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    load_model_path = args.load_model_path
    model = model.to(DEVICE)
    if load_model_path is not None:
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_model_path}.")
    print(f"Model is on device {next(model.parameters()).device}")
    model = model.train()
    
    # Freeze layers for finetuning
    freeze_model_start_layer = args.freeze_model_start_layer
    model = freeze_model_layers(model, freeze_model_start_layer)

    # Initialize loss function
    print("------------------------------Initializing Loss Function------------------------------")
    print("Args loss type", args.loss_type)
    use2d3d = args.use2d3d
    intermediate_weight = args.intermediate_weight
    if args.loss_type == "combo":
        criterion = ComboLoss(alpha=args.alpha, ce_ratio=args.ce_ratio)
        print("using combo loss", "alpha:", args.alpha, "ce_ratio", args.ce_ratio)
    elif args.loss_type == "focalt":
        if use2d3d:
            print("using focal tverskys loss with 2d3d")
            criterion = FocalTverskyLossWith2d3d(alpha=args.alpha, beta=args.beta, gamma=args.gamma, intermediate_weight=intermediate_weight)
        else:
            criterion = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            print("using focal tverskys loss")
        print("alpha:", args.alpha, "beta:", args.beta, "gamma:", args.gamma)
    elif args.loss_type == "dicebce":
        criterion = DiceBCELoss()
        print("using dicebce loss")
    elif args.loss_type == "focal":
        alpha = torch.Tensor([args.alpha, 1-args.alpha]).to(DEVICE)
        if use2d3d:
            print("using focal loss with 2d3d")
            criterion = FocalLossWith2d3d(alpha=args.alpha, gamma=args.gamma, intermediate_weight=intermediate_weight)
        else:
            criterion = FocalLoss(alpha=alpha, gamma=args.gamma, device=DEVICE)
            print(f"using focal loss")
        print("alpha:", args.alpha, "gamma:", args.gamma)
    else:
        criterion = DiceLoss()
        print("using dice loss")
    print("Loss function initialized.")

    print("------------------------------TRAINING------------------------------")
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
          loss_folder=loss_folder,
          num_predictions_to_log=args.num_predictions_to_log,
          depth=args.depth,
          height=args.height,
          width=args.width,
          use2d3d=False,
          model2d=True)