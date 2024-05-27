import os
import cv2
import numpy as np;
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# import segmentation_models_pytorch as smp
# import albumentations as album
import joblib

from typing import Tuple, List
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from collections import OrderedDict
from PIL import Image
import re

# import segmentation_models_pytorch.utils.metrics

class CaImagesDataset(torch.utils.data.Dataset):

    """Calcium imaging images dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            dataset_dir, 
            augmentation=None, 
            preprocessing=None,
            image_dim = (512, 512),
            mask_neurons=None,
            mask_mito=None,
            split=0 #0 for train, 1 for valid
    ):
        
        prefix = "train" if not split else "valid"
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_imgs"))) if "DS" not in image_id]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_gts"))) if "DS" not in image_id]

        if mask_neurons:
            assert "SEM_dauer_2_image_export_" in os.listdir(images_dir)[0], "illegal naming, feds on the way"
            self.neuron_paths = [os.path.join(os.path.join(dataset_dir, f"{prefix}_neuro"), neuron_id.replace("SEM_dauer_2_image_export_", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_")) for neuron_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_imgs")))]
        else: self.neuron_paths = None

        if mask_mito:
            self.mito_mask = [os.path.join(os.path.join(dataset_dir, f"{prefix}_mito"), neuron_id) for neuron_id.replace("png", "tiff") in sorted(os.path.join(dataset_dir, f"{prefix}_imgs"))]
        else:
            self.mito_mask=None
            
        self.augmentation = augmentation 
        self.preprocessing = preprocessing
        self.image_dim = image_dim
    
    def __getitem__(self, i):
        
        # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
        try:
            image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 
        except Exception as e:
            print(self.image_paths[i])
            raise Exception(self.image_paths[i])
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 0 or 255

        # make sure each pixel is 0 or 255
        
        mask_labels, counts = np.unique(mask, return_counts=True)
        # if (len(mask_labels)>2):
        #     print("More than 2 labels found for mask")
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        
        mask_ref = mask.copy()
        # apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        # img_size = 512
        # width, height = self.image_dim
        # max_dim = max(img_size, width,    )
        # pad_left = (max_dim-width)//2
        # pad_right = max_dim-width-pad_left
        # pad_top = (max_dim-height)//2
        # pad_bottom = max_dim-height-pad_top
        # _transform.append(transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), 
        #                                 padding_mode='edge'))
        # _transform.append(transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size=(img_size, img_size)))  

        mask = transforms.Compose(_transform)(mask)
        if len(mask_labels) == 1:
            mask[:] = 0
        else:
            mask[mask != 0] = 1
        ont_hot_mask = mask
        # ont_hot_mask = F.one_hot(mask.long(), num_classes=2).squeeze().permute(2, 0, 1).float()

        _transform_img = _transform.copy()
        # _transform_img.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        image = transforms.Compose(_transform_img)(image)

        #get the coresponding neuron mask
        if self.neuron_paths: neurons = cv2.cvtColor(cv2.imread(self.neuron_paths[i]), cv2.COLOR_BGR2GRAY)
        if self.mito_mask: mitos = np.array(Image.open(self.mito_mask[i]))
        
            
        return image, ont_hot_mask, neurons == 0 if self.neuron_paths else [], mitos if self.mito_mask else []
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class SectionsDataset(torch.utils.data.Dataset):
    
    def __init__(
            self, 
            dataset_dir 
            augmentation=None, 
            preprocessing=None,
            image_dim = (512, 512),
            chain_length=4,
            mask_neurons=None,
            mask_mito=None,
            split=0 #0 if train 1 if valid
    ):    
        prefix = "train" if not split else "valid"
        images_dir = os.path.join(dataset_dir, f"{prefix}_imgs")
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(os.path.join(dataset_dir, f"{prefix}_gts"), image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_gts")))]

        if mask_neurons:
            assert "SEM_dauer_2_image_export_" in os.listdir(images_dir)[0], "illegal naming, feds on the way"
            self.neuron_paths = [path.replace("imgs", "neuro") for path in self.image_paths]
        else: self.neuron_paths = None

        # if mask_mito:
        #     self.mito_mask = [os.path.join(mask_mito, neuron_id) for neuron_id in sorted(os.listdir(images_dir))]
        # else:
        #     self.mito_mask=None
        self.mito_mask = None
            
        self.augmentation = augmentation 
        self.preprocessing = preprocessing
        self.image_dim = image_dim
        self.chain_length = chain_length

        # self.make_chain_paths()
        print("Dataset has {} samples".format(len(self.image_paths)))

    def make_chain_paths(self):
        chain_img_paths, chain_seg_paths, chain_neuron_paths, chain_mito_paths = [], [], [] if self.neuron_paths else None, [] if self.neuron_paths else None
        for i in range(len(self.image_paths)):
            imgs, segs, neurons, mitos = [], [], [], []
            img_path, seg_path = self.image_paths[i], self.mask_paths[i]
            layer = re.findall(r's\d\d\d_', img_path)[0]

            skip = False
            for j in range(self.chain_length):
                next_layer_1 = "s" + ("00" if int(layer[1:-1]) < 9-j else "0") + str(int(layer[1:-1])+j) + "_"
                img_f_1 = img_path.replace(layer, next_layer_1)
                seg_f_1 = seg_path.replace(layer, next_layer_1)
                if self.neuron_paths:
                    neuron_path = self.neuron_paths[i]
                    neurons_f_1 = neuron_path.replace(layer, next_layer_1)
                if self.mito_mask:
                    mito_path = self.mito_mask[i]
                    mito_f_1 = mito_path.replace(layer, next_layer_1)

                if not os.path.isfile(img_f_1) or not os.path.isfile(seg_f_1) or (self.neuron_paths is not None and not os.path.isfile(neurons_f_1)) \
                    or (self.mito_mask is not None and not os.path.isfile(mito_f_1)) : 
                    skip = True
                    break
                # print("Not skip!")
                imgs.append(img_f_1)
                segs.append(seg_f_1)
                if self.neuron_paths: neurons.append(neurons_f_1)
                if self.mito_mask: mitos.append(mito_f_1)

            if skip: continue

            chain_img_paths.append(imgs)
            chain_seg_paths.append(segs)

            if self.neuron_paths: chain_neuron_paths.append(neurons)
            if self.mito_mask: chain_mito_paths.append(mitos)  
            # assert len(sum(chain_img_paths, start=[]))%3 == 0

        self.image_paths, self.mask_paths = chain_img_paths, chain_seg_paths
        if self.neuron_paths: self.neuron_paths = chain_neuron_paths
        if self.mito_mask: self.mito_mask = chain_mito_paths

    def __getitem__(self, i):

        # directory is arranged as before, current, future1, future2

        images = sorted(os.listdir(self.image_paths[i]))
        masks = sorted(os.listdir(self.mask_paths[i]))
        

        img = []
        ns = []
        if self.neuron_paths: neurons = sorted(os.listdir(self.neuron_paths[i]))
        if self.mito_mask: mitos = []

        mask = cv2.cvtColor(cv2.imread(os.path.join(self.mask_paths[i], masks[1])), cv2.COLOR_BGR2GRAY)

        for j in range(self.chain_length):
            im = cv2.cvtColor(cv2.imread(os.path.join(self.image_paths[i], images[j])), cv2.COLOR_BGR2GRAY)
            img.append(im)
            if self.neuron_paths:
                neuron = cv2.cvtColor(cv2.imread(os.path.join(self.neuron_paths[i], neurons[j])), cv2.COLOR_BGR2GRAY)
                ns.append(neuron)
        
        image = np.stack(img, axis=0)
        if self.neuron_paths: neurons = np.stack(ns, axis=0)
        mitos=[]
        

        # make sure each pixel is 0 or 255
        
        mask_labels = np.unique(mask)
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        
        
        # apply augmentations
        flippant = random.random()
        if self.augmentation and flippant < 0.25:
            if flippant < 0.55:
                #random rotation plus gaussian blur
                sample = self.transform1(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            elif flippant < 0.85:
                #horizontal and vertical flip + gaussian blur
                sample = self.transform2(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                #everything
                sample = self.transform3(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        _transform = []
        _transform.append(transforms.ToTensor())

        mask = transforms.Compose(_transform)(mask)
        if len(mask_labels) == 1:
            mask[:] = 0
        else:
            mask[mask != 0] = 1
        ont_hot_mask = mask

        _transform_img = _transform.copy()
        image = transforms.Compose(_transform_img)(image)

        image = torch.permute(image, (1, 2, 0))  

        # ont_hot_mask = torch.permute(ont_hot_mask, (1, 2, 0))    
            
        return image.unsqueeze(0), ont_hot_mask, torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], torch.from_numpy(mitos[np.newaxis, :]) if self.mito_mask else []
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, three=False):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DownBlock(nn.Module):
    """Double Convolution followed by Max Pooling"""
    def __init__(self, in_channels, out_channels, three=False):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, three=three)
        self.down_sample = nn.MaxPool2d(2, stride=2) if not three else nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    """Up Convolution (Upsampling followed by Double Convolution)"""
    def __init__(self, in_channels, out_channels, up_sample_mode, three=False):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            if three: self.up_sample = nn.ConvTranspose3d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
            else: self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True, three=three)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels, three=three)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False):
        """Initialize the UNet model"""
        super(UNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width)    
        x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        if self.three: x = x.squeeze(-3)   
        x, skip3_out = self.down_conv3(x) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.three: 
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class SplitUNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False):
        """Initialize the UNet model"""
        super(SplitUNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512) # 256 input channels --> 512 output channels
        self.down_conv5 = DownBlock(1024, 1024)
        self.down_conv6 = DownBlock(1024, 1024)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1) if not three else nn.Conv3d(64, 1, kernel_size=1)

        self.flats = nn.Sequential(OrderedDict([
            ('flat1', nn.Linear((3 if three else 1) * 4096, 1024)),
            ('relu1', nn.ReLU()),
            ('flat2', nn.Linear(1024, 64)),
            ('relu2', nn.ReLU()),
            ('flat3', nn.Linear(64, 1))
        ]))

        self.once = True

    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width)
        if self.once: print("Step 1 shape {} and skipout {}".format(x.shape, skip1_out.shape))
        x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        if self.once:print("Step 2 shape {} and skipout {}".format(x.shape, skip2_out.shape))
        x, skip3_out = self.down_conv3(x) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        if self.once:print("Step 3 shape {} and skipout {}".format(x.shape, skip3_out.shape))
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        if self.once:print("Step 4 shape {} and skipout {}".format(x.shape, skip4_out.shape))
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        if self.once:print("Step 5 shape {}".format(x.shape))
    
        x_, _ = self.down_conv5(x)
        x_, _ = self.down_conv6(x_)
        x_ = self.flats(x_.view(x_.shape[0], -1)) # flattent and pass into 

        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        if self.once:print("Step 6 shape {}".format(x.shape))
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.once:print("Step 7 shape {}".format(x.shape))
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        if self.once:print("Step 8 shape {}".format(x.shape))
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        if self.once:print("Step 9 shape {}".format(x.shape))
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        if self.once:print("Step 10 shape {}".format(x.shape))
        if self.once: self.once = False
        return x, x_
    
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, device=torch.device("cpu")):
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.device = device
        self.alpha = alpha.to(device)
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[]):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        targets = targets.to(torch.int64)
        loss = self.alpha[targets.view(targets.shape[0], -1)].reshape(targets.shape) * (1-pt) ** self.gamma * bce_loss
        if mito_mask != []:
            #first modify loss_mask, neuron_mask is always on.
            loss_mask = loss_mask | mito_mask
            # factor = 1
            # loss = loss * (1 + (mito_mask * factor))#weight this a bit more. 
        if loss_mask != []: 
            #better way? TODO: get rid of this if statement
            if len(loss.shape) > len(loss_mask.shape): loss = loss * loss_mask.unsqueeze(-1)
            else: loss = loss * loss_mask # remove everything that is a neuron body, except ofc if the mito_mask was on. 

        return loss.mean() 

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[]):
        #get the probabilities:
        probs = nn.Sigmoid()(inputs)
        n1 = probs * targets
        inv_targets = 1 - targets

        d = probs + targets

        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        
        if loss_mask != []:
            if len(targets.shape) > len(loss_mask.shape): loss_mask.unsqueeze(-1)
            # loss_mask = loss_mask.view(loss_mask.shape[0], -1)
            inv_targets[loss_mask] = 0
            d *= (loss_mask == 0)
            d += (loss_mask *2) # eqv to d[loss_mask]=2

        n2 = (1-probs) * inv_targets
    
        # no need to use masks for balancing, coz of the loss fn here

        n1 = n1.view(n1.shape[0], -1)
        n2 = n2.view(n2.shape[0], -1)
        d =  d.view(d.shape[0], -1)

        return torch.mean(1 - (torch.sum(n1, dim=-1) + self.eps)/(torch.sum(d, dim=-1) + self.eps)
                           - (torch.sum(n2, dim=-1) + self.eps)/(torch.sum(2-d, dim=-1) + self.eps))

class SSLoss(nn.Module):
    def __init__(self, eps=1e-6, lamda=0.05):
        super(SSLoss, self).__init__()
        self.eps = eps
        self.lamda = 0.05

    def forward(self, inputs, targets, loss_mask=[], mito_mask=[]):
        inputs = nn.Sigmoid()(inputs)
        targets, inputs = targets.view(targets.shape[0], -1), inputs.view(inputs.shape[0], -1)
        inv_targets = 1-targets
        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        
        if loss_mask != []:
            if len(targets.shape) > len(loss_mask.shape): loss_mask.unsqueeze(-1)
            loss_mask = loss_mask.view(loss_mask.shape[0], -1)
            inv_targets[loss_mask] = 0

        return torch.mean(self.lmda * torch.sum(torch.pow((targets - inputs), 2) * targets, dim=-1)/(torch.sum(targets, dim=-1) + self.eps) +
                           (1-self.lamda) * torch.sum(torch.pow((targets - inputs), 2) * inv_targets, dim=-1)/ (torch.sum(inv_targets, dim=-1) + self.eps))

class GenDLoss(nn.Module):
    def __init__(self):
        super(GenDLoss, self).__init__()
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[]):
        inputs = nn.Sigmoid()(inputs)
        targets, inputs = targets.view(targets.shape[0], -1), inputs.view(inputs.shape[0], -1)

        inputs = torch.stack([inputs, 1-inputs], dim=-1)
        targets = torch.stack([targets, 1-targets], dim=-1)

        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        
        if loss_mask != []:
            if len(targets.shape) > len(loss_mask.shape): loss_mask.unsqueeze(-1)
            loss_mask = loss_mask.view(loss_mask.shape[0], -1)
            targets *= loss_mask.unsqueeze(-1)
            inputs *= loss_mask.unsqueeze(-1)#0 them out in both masks

        weights = 1 / (torch.sum(torch.permute(targets, (0, 2, 1)), dim=-1).pow(2)+1e-6)
        targets, inputs = torch.permute(targets, (0, 2, 1)), torch.permute(inputs, (0, 2, 1))

        # print(torch.nansum(weights * torch.nansum(targets * inputs, dim=-1), dim=-1))
        # print(weights)

        return torch.nanmean(1 - 2 * torch.nansum(weights * torch.nansum(targets * inputs, dim=-1), dim=-1)/\
                          torch.nansum(weights * torch.nansum(targets + inputs, dim=-1), dim=-1))

def get_training_augmentation():
    """ Add augmentation to the training data. Crop it to 256, 256 and flip it horizontally, vertically or rotate it by 90 degrees.
    
    Returns:
        album.Compose: Composed augmentation functions
    """
    train_transform = [    
        album.RandomCrop(height=256, width=256, always_apply=True), # crop it to 256, 256
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():   
    """ Add augmentation to the validation data. Add padding to make it 1536, 1536. 
    
    Returns:
        album.Compose: Composed augmentation functions
    """
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0), # crop it to 1536, 1536
    ]
    return album.Compose(test_transform)


def get_preprocessing(preprocessing_fn=None, image_dim=(512, 512)):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    _transform.append(transforms.ToTensor())

    img_size = 512
    width, height = image_dim
    max_dim = max(img_size, width, height)
    pad_left = (max_dim-width)//2
    pad_right = max_dim-width-pad_left
    pad_top = (max_dim-height)//2
    pad_bottom = max_dim-height-pad_top
    _transform.append(transforms.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), 
                                    padding_mode='edge'))
    _transform.append(transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size=(img_size, img_size)))       
    if preprocessing_fn:
        _transform.append(preprocessing_fn)

    return transforms.Compose(_transform)

def visualize(**images):
    """
    Plot images in one row
    images: List of images in the form (width, height, channels)
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    # for each item in images, it has an index, name, and image
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()
    

def find_centroids(segmented_img):
    centroids = []
    cont, hierarchy = cv2.findContours(segmented_img, 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))
    
    return centroids

# Center crop padded image / mask to original image dims
def crop_image(image, target_image_dims):

    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

def get_subset(dataset, subset_size):
    """Get a subset of the dataset
    
    Args:
        dataset (CaImagesDataset): dataset to get subset from
        subset_size (int): size of the subset
        
    Returns:
        subset (CaImagesDataset): subset of the dataset
    """
    # get a random subset of the dataset
    subset = torch.utils.data.Subset(dataset, random.sample(range(len(dataset)), subset_size))
    return subset
