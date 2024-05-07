if __name__ == '__main__':    

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
    import sys

    ## DEFINE UNET MODEL

    x_train_dir=r"E:\Mishaal\GapJunction\small_data\original\train"
    y_train_dir=r"E:\Mishaal\GapJunction\small_data\ground_truth\train"

    x_valid_dir=r"E:\Mishaal\GapJunction\small_data\original\valid"
    y_valid_dir=r"E:\Mishaal\GapJunction\small_data\ground_truth\valid"

    x_test_dir=r"E:\Mishaal\GapJunction\small_data\original\test"
    y_test_dir=r"E:\Mishaal\GapJunction\small_data\ground_truth\test"

    model_folder = r"E:\Mishaal\GapJunction\models"
    sample_preds_folder = r"E:\Mishaal\GapJunction\results"

    height, width = cv2.imread(os.path.join(x_train_dir, os.listdir(x_train_dir)[0])).shape[:2]

    # Get train and val dataset instances
    augmentation = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.4)
    ])

    train_dataset = CaImagesDataset(
        x_train_dir, y_train_dir, 
        preprocessing=None,
        image_dim = (width, height), augmentation=augmentation
    )
    valid_dataset = CaImagesDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=None,
        preprocessing=None,
        image_dim = (width, height)
    )

    SEED = 12
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)
    print("Data loaders created.")

    WANDB_API_KEY = "42a2147c44b602654473783bde1ecd15579cc313"
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    TRAINING = True
    NEW = True
    model_name = "model5"
    TESTING = False
    VALIDATE= False
    def sigint_handler(sig, frame):
        global table
        if table is not None and TRAINING:
            print("logging to WANDB")
            wandb.log({"Table" : table})
            joblib.dump(model, os.path.join(model_folder, f"{model_name}.pk1"))
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    if TRAINING:
        lr = 0.001
        epochs = 30
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

        if NEW:
            model = UNet().to(DEVICE)
        else:
            model = joblib.load(os.path.join(model_folder, "model4.pk1")) # load model

        class_labels = {
        1: "background",
        0: "gapjunction",
        }
        
        smushed_labels = None
        for i in range(len(train_dataset)):
            if smushed_labels is None: smushed_labels = train_dataset[i][1].to(torch.int64)
            else: smushed_labels = torch.concat([smushed_labels, train_dataset[i][1].to(torch.int64)])
        class_counts = torch.bincount(smushed_labels.flatten())
        total_samples = len(train_dataset) * 512 * 512
        w1, w2 = 1/(class_counts[0]/total_samples), 1/(class_counts[1]/total_samples)
        cls_weights = torch.Tensor([w1, w2/9])

        print(cls_weights)

        print("Starting training...")
        start = time.time()
        alpha = 0.25  
        gamma = 3
        criterion = FocalLoss(alpha=cls_weights, device=DEVICE)#torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        decayed_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        loss_list = [] 
        table = wandb.Table(columns=['ID', 'Image'])

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                print("Progress: {:.2%}".format(i/len(train_loader)))
                inputs, labels = data # (inputs: [batch_size, 1, 512, 512], labels: [batch_size, 1, 512, 512])
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                pred = model(inputs)
                loss = criterion(pred, labels) # calculate loss (binary cross entropy)
                # p_t = torch.exp(-bce_loss)
                # focal_loss = alpha* (1 - p_t) ** gamma * bce_loss
                # loss = focal_loss.mean()
                loss.backward() # calculate gradients (backpropagation)
                optimizer.step() # update model weights (values for kernels)
                print(f"Step: {i}, Loss: {loss}")
                loss_list.append(loss)
                wandb.log({"loss": loss})
            
            epoch_non_empty = False

            for i, data in enumerate(valid_loader):
                valid_inputs, valid_labels = data
                valid_inputs, valid_labels = valid_inputs.to(DEVICE), valid_labels.to(DEVICE)
                valid_pred = model(valid_inputs)
                valid_loss = criterion(valid_pred, valid_labels) # calculate loss (binary cross entropy)
                # p_t = torch.exp(-bce_loss)
                # focal_loss = alpha* (1 - p_t) ** gamma * bce_loss
                # valid_loss = focal_loss.mean()
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
                uniques = np.unique(torch.round(nn.Sigmoid()(valid_pred[0].squeeze(0))).detach().cpu().numpy())
                if len(uniques) == 2:
                    if not epoch_non_empty:
                        epoch_non_empty = True
                        print("UNIQUE OUTPUTS!")
                else:
                    epoch_non_empty = False
            decayed_lr.step(valid_loss)

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

    if TESTING:
        sample_train_folder = sample_preds_folder+"\\train_res"
        model = joblib.load(os.path.join(model_folder, "model5_epoch17.pk1"))
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

        sample_val_folder = sample_preds_folder+"\\valid_res"
        for i in tqdm(range(len(valid_dataset))):
            image, gt_mask = train_dataset[i] # image and ground truth from test dataset
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
        
    exit()

    LOAD = False # True if loading a model, False if creating a new model
    TRAINING = True # True if training, False if testing
    load_num = 3
    save_num = 3
    target_width = 454 # change to max width of images in dataset (make sure dividable by 2)
    target_height = 546 # change to max height of images in dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    img, mask = train_dataset[0]

    print(img.shape, mask.shape)
    # exit()
    #TRAIN MODEL
    if TRAINING:
        if LOAD:
            model = joblib.load(f'{model_folder}/model{load_num}.pkl')
        else:
            model = UNet()
        # loss = smp.utils.losses.DiceLoss()
        # metrics = [
        #     smp.utils.metrics.IoU(threshold=0.5),
        # ]
        # optimizer = torch.optim.Adam([ 
        #     dict(params=model.parameters(), lr=0.00008),
        # ])

        # # define data loaders for training and validation sets
        # train_epoch = smp.utils.train.TrainEpoch(
        #     model, 
        #     loss=loss, 
        #     metrics=metrics, 
        #     optimizer=optimizer,
        #     device=DEVICE,
        #     verbose=True,
        # )

        # valid_epoch = smp.utils.train.ValidEpoch(
        #     model, 
        #     loss=loss, 
        #     metrics=metrics, 
        #     device=DEVICE,
        #     verbose=True,
        # )
        
        print("Starting training...")
        EPOCHS = 1
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        train_logs_list, valid_logs_list = [], [] # train and valid logs

        for i in range(0, EPOCHS):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            j=0
            for img, mask in train_loader:
                print("Progress: {:.2%}".format(j/len(train_loader)))
                pred = model(img)
                loss = criterion(pred, mask)
                loss.backward()
                optimizer.step()
                j+=1
            
            if i % 1 == 0:
                print(f"Epoch: {i}, Loss: {loss}")
            # train_logs = train_epoch.run(train_loader)
            # valid_logs = valid_epoch.run(valid_loader)
            # train_logs_list.append(train_logs)
            # valid_logs_list.append(valid_logs)
        
        joblib.dump(model, f'{model_folder}/model{save_num}.pkl')
    else:
        model = joblib.load(f'{model_folder}/model{load_num}.pkl')

    ## TEST MODEL
    test_dataset = CaImagesDataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn=None),   
    )

    test_dataloader = DataLoader(test_dataset)

    random_idx = random.randint(0, len(test_dataset)-1)
    image, mask = test_dataset[random_idx]
    plt.imshow(image.squeeze().to_numpy(), cmap='gray')
    plt.show()
    plt.imshow(mask.squeeze().to_numpy(), cmap='gray')
    plt.show()



    for idx in range(2):
        image, gt_mask = test_dataset[random_idx] # image and ground truth from test dataset
        print(gt_mask.shape)
        print(gt_mask) #CHW (2, H, W)
        image_vis = crop_image(np.transpose(test_dataset[random_idx][0].astype('uint8'), (1, 2, 0)), (target_height, target_width, 3)) # image for visualization
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        # Predict test image
        pred_mask = model(x_tensor)
        print(pred_mask.shape)
        print(pred_mask)


        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask,(1,2,0))
        print(pred_mask.shape)
        # Get prediction channel corresponding to calcium
        pred_calcium_heatmap = pred_mask[:,:,class_names.index('calcium')]
        print(pred_calcium_heatmap.shape)
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), class_rgb_values), (target_height, target_width, 1))

        # Convert gt_mask from `CHW` format to `HWC` format
        gt_mask = np.transpose(gt_mask,(1,2,0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), class_rgb_values), (target_height, target_width, 3))

        print(pred_mask)
        print(pred_mask.shape)
        plt.imshow(pred_mask)
        plt.show()

        # cv2.imwrite(
        #     os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), 
        #     np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1]
        #     )  
        # visualize(
        #     original_image = image_vis,
        #     ground_truth_mask = gt_mask,
        #     predicted_mask = pred_mask,
        #     predicted_building_heatmap = pred_calcium_heatmap
        # )

    # EVALUATE MODEL
    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(test_dataloader)
    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

        
        