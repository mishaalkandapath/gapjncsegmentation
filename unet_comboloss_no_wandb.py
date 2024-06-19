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
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("seed:", seed)


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
    # data_dir = args.data_dir
    # if not os.path.exists(data_dir): print(f"Data directory {data_dir} does not exist.")

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
    
    img_dir_list = args.img_dir_list
    mask_dir_list = args.gt_dir_list
    valid_img_dir_list = args.valid_img_dir_list
    valid_mask_dir_list = args.valid_gt_dir_list
    if img_dir_list is not None:
        print("Setting up from lists")
        train_dataset, train_loader = setup_datasets_and_dataloaders_from_lists(img_dir_list=img_dir_list, mask_dir_list=mask_dir_list, batch_size=batch_size, num_workers=num_workers, augment=args.augment, shuffle=True)
        valid_dataset, valid_loader = setup_datasets_and_dataloaders_from_lists(img_dir_list=valid_img_dir_list, mask_dir_list=valid_mask_dir_list, batch_size=1, num_workers=num_workers, augment=False, shuffle=False)
    elif args.train_cellmask_dir is not None:
        print("Using cell mask -- 2 class prediction")
        train_dataset, train_loader = setup_datasets_and_dataloaders_withmemb(args.train_x_dir, args.train_y_dir, args.train_cellmask_dir, batch_size, num_workers, args.augment, use3classes=args.pred3classes, shuffle=True)
        valid_dataset, valid_loader = setup_datasets_and_dataloaders_withmemb(args.valid_x_dir, args.valid_y_dir, args.valid_cellmask_dir, batch_size, num_workers, False, use3classes=args.pred3classes, shuffle=False)
        # train_dataset, valid_dataset, train_loader, valid_loader = setup_datasets_and_dataloaders_withmemb(data_dir, args.cellmask_dir, batch_size, num_workers, augment=args.augment)
    else:
        train_dataset, train_loader = setup_datasets_and_dataloaders(args.train_x_dir, args.train_y_dir, batch_size, num_workers, args.augment, shuffle=True)
        valid_dataset, valid_loader = setup_datasets_and_dataloaders(args.valid_x_dir, args.valid_y_dir, batch_size, num_workers, False, shuffle=False)
        # train_dataset, valid_dataset, train_loader, valid_loader = setup_datasets_and_dataloaders(data_dir, batch_size, num_workers, augment=args.augment)
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
    model = model.train()
    
    # Freeze layers for finetuning
    freeze_model_start_layer = args.freeze_model_start_layer
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
          num_predictions_to_log=args.num_predictions_to_log,
          depth=args.depth,
          height=args.height,
          width=args.width)
