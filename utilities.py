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
import torchvision.transforms.v2 as v2
# import segmentation_models_pytorch as smp
# import albumentations as album
import joblib

from typing import Tuple, List
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torchvision.ops import DropBlock2D, DropBlock3D
from collections import OrderedDict
from PIL import Image
import re, math
import pickle as p

from resnet import ResNet, BasicBlock

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
        images_dir = os.path.join(dataset_dir, f"{prefix}_imgs")
        masks_dir = os.path.join(dataset_dir, f"{prefix}_gts")
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_imgs"))) if "DS" not in image_id]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_gts"))) if "DS" not in image_id]

        if mask_neurons:
            assert "SEM_dauer_2_image_export_" in os.listdir(images_dir)[0], "illegal naming, feds on the way"
            self.neuron_paths = [os.path.join(os.path.join(dataset_dir, f"{prefix}_neuro"), neuron_id.replace("SEM_dauer_2_image_export_", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_").replace(".png.png", ".png")) for neuron_id in sorted(os.listdir(os.path.join(dataset_dir, f"{prefix}_imgs")))]
        else: self.neuron_paths = None

        if mask_mito:
            self.mito_mask = [os.path.join(os.path.join(dataset_dir, f"{prefix}_mito"), neuron_id.replace("png", "tiff") if not "tiny" in dataset_dir else neuron_id) for neuron_id in sorted(os.listdir(images_dir))]
            assert len(self.mito_mask) == len(self.image_paths), f"{len(self.mito_mask)} {len(self.image_paths)}"
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
            image = v2.RandomAutocontrast()(image)
        
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
        
            
        return image, ont_hot_mask,torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], mitos if self.mito_mask else []
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class SectionsDataset(torch.utils.data.Dataset):
    
    def __init__(
            self, 
            dataset_dir ,
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

        for j in range(4):
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

class DebugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, aug=False):
        images = os.path.join(dataset_dir, "imgs")
        gts = os.path.join(dataset_dir, "gts")
        mems = os.path.join(dataset_dir, "neurons")

        self.mask_paths = []
        self.image_paths = []
        self.neuron_paths = []
        names = []

        print("--Filtering")
        # for image_id in tqdm(sorted(os.listdir(images))):
        #     if "DS" in image_id:
        #         continue
        #     image_id = os.path.join(gts, image_id)
        #     gt = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2GRAY)
        #     gt[gt == 2] = 0
        #     gt[gt == 15] = 0
        #     if len(np.unique(gt)) < 2: continue
        #     image_name = os.path.split(image_id)[-1]
        #     names.append(image_name)
        #     self.image_paths.append(os.path.join(images, image_name))
        #     self.neuron_paths.append(os.path.join(mems, image_name))
        #     self.mask_paths.append(os.path.join(gts, image_id))
        with open("/home/mishaalk/projects/def-mzhen/mishaalk/gapjncsegmentation/names.p", "rb") as  f:
            names = p.load(f)
        for image_name in names:
            self.image_paths.append(os.path.join(images, image_name))
            self.neuron_paths.append(os.path.join(mems, image_name))
            self.mask_paths.append(os.path.join(gts, image_name))
        
        self.augmentation = aug
        

        
    def __getitem__(self, i):

        # directory is arranged as before, current, future1, future2
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY)
        neurons = cv2.cvtColor(cv2.imread(self.neuron_paths[i]), cv2.COLOR_BGR2GRAY) 
        

        # make sure each pixel is 0 or 255
        # print(np.unique(mask))
        
        mask_labels = np.unique(mask)
        mask[mask == 0] = 0
        mask[mask == 255] = 1
        
        
        # apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
            image = v2.RandomAutocontrast()(image)
        
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

        # image = torch.permute(image, (1, 2, 0))     
        if len(image.shape) >= 5: print(image.shape)
        # print( torch.from_numpy(neurons[np.newaxis, :]).unsqueeze(0).shape)
            
        return image, ont_hot_mask, torch.from_numpy(neurons[np.newaxis, :]) == 0 if self.neuron_paths else [], []

    def __len__(self):
        # return length of 
        return len(self.image_paths)
    



class TestDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_dir ,
            image_dim = (512, 512),
            td=False,
            membrane=False
    ):      
        self.image_paths = [os.path.join(dataset_dir, image_id) for image_id in sorted(os.listdir(dataset_dir)) if "DS" not in image_id]
        self.mask_paths = [os.path.join(dataset_dir.replace("imgs", "gts"), image_id) for image_id in sorted(os.listdir(dataset_dir)) if "DS" not in image_id]
        if membrane:
            self.membrane_paths = [os.path.join(dataset_dir, image_id) for image_id in sorted(os.listdir(dataset_dir)) if "DS" not in image_id]
        else: self.membrane_paths = None
        self.td = td

    def __getitem__(self, i):
        # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
        if not self.td:
            try:
                image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2GRAY) # each pixel is 
                mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
                if self.membrane_paths: memb = cv2.cvtColor(cv2.imread(self.membrane_paths[i]), cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(self.image_paths[i])
                raise Exception(self.image_paths[i])
        else:
            images = sorted(os.listdir(self.image_paths[i]))
            img, mem = [], []
            for j in range(4):
                im = cv2.cvtColor(cv2.imread(os.path.join(self.image_paths[i], images[j])), cv2.COLOR_BGR2GRAY)
                if self.membrane_paths: 
                    memb = cv2.cvtColor(cv2.imread(os.path.join(self.membrane_paths[i], images[j])), cv2.COLOR_BGR2GRAY)
                    mem.append(memb)
                img.append(im)
            
            image = np.stack(img, axis=0)
            memb = np.stack(mem, axis=0)

        _transform = []
        _transform.append(transforms.ToTensor()) 
        if self.membrane_paths: memb = transforms.Compose(_transform)(memb)
        _transform_img = _transform.copy()
        # _transform_img.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        image = transforms.Compose(_transform_img)(image)
        mask = transforms.Compose(_transform_img)(mask)
        if self.td: 
            image = torch.permute(image, (1, 2, 0)) 
            if self.membrane_paths: memb = torch.permute(memb, (1, 2, 0)) 
        else: 
            image = image.squeeze(0) 
            memb = memb.squeeze(0)
            mask = mask.squeeze(0)


        if image.shape[-1] != 512: 
            image = torch.zeros(image.shape[:-1]+(512,))
            mask = torch.zeros(image.shape[:-1]+(512,))
            if self.membrane_paths: memb = torch.zeros(memb.shape[:-1]+(512,))
        if image.shape[-2] != 512: 
            image = torch.zeros(image.shape[:-2]+(512,512))
            mask = torch.zeros(image.shape[:-1]+(512,))
            if self.membrane_paths: memb = torch.zeros(memb.shape[:-2]+(512,512))

        return image.unsqueeze(0), mask.unsqueeze(0), image.unsqueeze(0) if not self.membrane_paths else memb.unsqueeze(0), image.unsqueeze(0)
    def __len__(self):
        # return length of 
        return len(self.image_paths)

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, three=False, spatial=False, residual=False, dropout=0):
        super(DoubleConv, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.projection_add = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.final = nn.Sequential(
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            self.dropout,
        )
        self.spatial=spatial
        if spatial: 
            self.spatial_sample = PyramidPooling(levels=[2, 2, 4, 4, 4, 4, 4, 4,4, 4], td=three)
        self.residual = residual

    def forward(self, x_in):
        x = self.double_conv(x_in)
        # if self.residual: x = x + self.projection_add(x_in)
        # x = self.final(x)
        con_shape = x.shape
        # if self.spatial: # Spatial pyramidal pooling
        #     x = self.spatial_sample(x)
        #     x = x.reshape(con_shape)
        return x
    
class DownBlock(nn.Module):
    """Double Convolution followed by Max Pooling"""
    def __init__(self, in_channels, out_channels, three=False, spatial=True, dropout=0, residual=False):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, three=three, dropout=dropout, residual=residual)
        self.spatial = spatial
        if spatial: 
            self.spatial_sample = PyramidPooling(levels=[2, 2, 4, 4, 4, 4, 4, 4,4, 4], td=three)
        self.down_sample = nn.MaxPool2d(2, stride=2) if not three else nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        # if self.spatial: 
        #     x = self.spatial_sample(skip_out)
        #     x = x.reshape(skip_out.shape)
        #     down_out = self.down_sample(x)
        # else:   
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    """Up Convolution (Upsampling followed by Double Convolution)"""
    def __init__(self, in_channels, out_channels, up_sample_mode, kernel_size=2, three=False, dropout=0, residual=False):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            if three: self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(in_channels-out_channels, in_channels-out_channels, kernel_size=kernel_size, stride=2),
                nn.ReLU())       
            else: self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=kernel_size, stride=2),
                nn.ReLU())
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True, three=three)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels, three=three, residual=residual)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, attend=False, residual=False, scale=False, spatial=False, dropout=0):
        """Initialize the UNet model"""
        super(UNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        self.dropout=dropout

        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three, spatial=False, residual=residual) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three, spatial=spatial, dropout=self.dropout, residual=residual) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256, spatial=spatial, dropout=self.dropout, residual=residual) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512, spatial=spatial, dropout=self.dropout, residual=residual) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024,spatial=spatial, dropout=self.dropout, residual=residual)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode, dropout=self.dropout, residual=residual) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode, dropout=self.dropout, residual=residual)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode, dropout=self.dropout, residual=residual)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        self.attend = attend
        if scale:
            self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling


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
            #attention_mode???
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class ResUNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, dropout=0):
        """Initialize the UNet model"""
        super(ResUNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        self.dropout=dropout

        # Downsampling Path
        self.resnet = ResNet(BasicBlock, [1, 4, 6, 3, 3], norm_layer = nn.BatchNorm2d if not three else nn.BatchNorm3d, three=three)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode, dropout=self.dropout, kernel_size=2) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode, kernel_size=2)

        #extra convs:
        self.m1 = nn.Sequential(
                nn.ConvTranspose2d(64, 64,kernel_size=2, stride=2),
                nn.ReLU())
        self.m2 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                nn.ReLU())

        self.add_conv1 = DoubleConv(64, 64)
        self.add_conv2 = DoubleConv(64, 64)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        # if scale:
        #     self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling


    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out, skip2_out, skip3_out, skip4_out = self.resnet(x)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        # if self.three: 
        #     #attention_mode???
        #     skip1_out = torch.mean(skip1_out, dim=2)
            # skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.m1(x)
        x = self.add_conv1(x)
        x = self.m2(x)
        x = self.add_conv2(x)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class MemResUNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, dropout=0):
        """Initialize the UNet model"""
        super(MemResUNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        self.dropout=dropout

        # Downsampling Path
        self.resnet = ResNet(BasicBlock, [1, 4, 6, 3, 3], norm_layer = nn.BatchNorm2d if not three else nn.BatchNorm3d, three=three, membrane=True)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode, dropout=self.dropout, kernel_size=2) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode, kernel_size=2)

        #extra convs:
        self.m1 = nn.Sequential(
                nn.ConvTranspose2d(64, 64,kernel_size=2, stride=2),
                nn.ReLU())
        self.m2 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                nn.ReLU())

        self.add_conv1 = DoubleConv(64, 64)
        self.add_conv2 = DoubleConv(64, 64)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        # if scale:
        #     self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling


    def forward(self, x, mem_x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)

        x, skip1_out, skip2_out, skip3_out, skip4_out = self.resnet((x, mem_x))
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.three: 
            #attention_mode???
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.m1(x)
        x = self.add_conv1(x)
        x = self.m2(x)
        x = self.add_conv2(x)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x


class MemUNet(nn.Module):
    """UNet Architecture with membrane feature information"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, attend=False, scale=False):
        """Initialize the UNet model"""
        super(MemUNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels
        self.down_conv1_mem = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2_mem = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels
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
        self.attend = attend
        if scale:
            self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling
        if attend:
            self.attention1 = nn.MultiheadAttention(512*512, 4, dropout=0.2)
            self.attention2 = nn.MultiheadAttention(256*256, 4, dropout=0.2)


    def forward(self, x, mem_x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width) 
        x_mem, skip1_out_mem = self.down_conv1_mem(mem_x)   
        x, skip2_out = self.down_conv2(x+x_mem) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        x_mem, skip2_out_mem = self.down_conv2_mem(x_mem)
        if self.three: x = x.squeeze(-3), x_mem.squeeze(-3)   
        x, skip3_out = self.down_conv3(x+x_mem) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.three: 
            #attention_mode???
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

    
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

        return torch.mean(2 - (torch.sum(n1, dim=-1) + self.eps)/(torch.sum(d, dim=-1) + self.eps)
                           - (torch.sum(n2, dim=-1) + self.eps)/(torch.sum(2-d, dim=-1) + self.eps))

class TverskyFocal(nn.Module):
    def __init__(self, eps=1e-6):
        super(TverskyFocal, self).__init__()
        self.eps = eps
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], recall=0.7, precision=0.3):
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

        f1 = probs * inv_targets
        f2 = (1-probs) * targets
    
        # no need to use masks for balancing, coz of the loss fn here

        n1 = n1.view(n1.shape[0], -1)
        n2 = n2.view(n2.shape[0], -1)
        d =  d.view(d.shape[0], -1)

        f1 = torch.sum(f1.view(n1.shape[0], -1))
        f2 = torch.sum(f2.view(n1.shape[0], -1))

        return torch.mean(2 - (torch.sum(n1, dim=-1) + self.eps)/(torch.sum(n1, dim=-1) + recall*f2 + precision*f1  + self.eps)
                           - (torch.sum(n2, dim=-1) + self.eps)/(torch.sum(n2, dim=-1) + precision*f2 + recall*f1 + self.eps))

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

class SpecialLoss(nn.Module):
    def __init__(self, bg_imp=0.2):
        super(GenDLoss, self).__init__()
        self.bg_imp = bg_imp
    
    def forward(self, predictions, target_centers, targets, pad_mask):
        _, _, center_rows, center_cols = torch.where(target_centers == 1)
        pad_tensors = torch.zeros((target_centers.shape[-2:]))
        pad_tensors[0, 0] = 1 # simply 

        target_centers[pad_mask] = pad_tensors # for now

        rows, cols = np.indices(target_centers.shape[-2:])
        rows, cols = torch.from_numpy(rows), torch.from_numpy(cols)
        #extend 
        rows, cols = rows.expand(predictions.size(0), len(center_rows)//predictions.size(0), -1, -1), cols.expand(predictions.size(0), len(center_rows)//predictions.size(0), -1, -1)
        center_rows, center_cols = center_rows.view(predictions.size(0), predictions.size(1), 1, 1), center_cols.view(predictions.size(0), predictions.size(1), 1, 1)
        
        squared_dist = (rows - center_rows) ** 2 + (cols - center_cols) ** 2
        pixel_importance = (targets) * 1/((squared_dist)+1e-6)
        pixel_importance[pad_mask] = 0 # reset everything that was purely pad

        pixel_importance = pixel_importance.sum(dim=1)
        importance_coeff = (~targets.sum(dim=1)) * 0.5
        pixel_importance += importance_coeff

        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.sum(dim=1).to(dtype=torch.float32), reduction="none")
        bce_loss *= pixel_importance

        return torch.mean(bce_loss)
        


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

class PyramidPooling(nn.Module):

    def __init__(self, levels, channels=1, mode="max", method='spatial', td=False):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param channels defines the number of "color" channels in the data (used to determine dimensionality)
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :param method defines whether spatial or temporal pyramid pooling is used

        :returns a tensor vector with shape [batch x channels x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super().__init__()
        assert all(isinstance(_, int) for _ in levels)
        assert all(_ > 0 for _ in levels)
        self.levels = levels
        self.channels = channels
        self.mode = mode

        assert method.lower() in ['spatial', 'temporal']
        self.method = method.lower()

    def get_output_size(self):
        out = 0
        for level in self.levels:
            out += level * (level if self.method == 'spatial' else 1)
        return out
    
    def forward(self, x, compress_single_dimensions=True):
        assert isinstance(x, torch.Tensor)
        assert 2 <= len(x.shape) <= 4, "input x must be 2 dimensional (1 sample, 1 channel), 3 dimensional (1 sample, n channels | n samples, 1 channel) or 4 dimensional (n samples, n channels)"
        if len(x.shape) == 2:
            n_samples = 1
            assert self.channels == 1, "2 dimensional input passed when self.channels == %d (implicit single channel passed with 2D)" % self.channels
        elif len(x.shape) == 3:
            if self.channels == 1:
                n_samples = x.shape[0]
            else:
                assert self.channels == x.shape[0], "%d channels specified but 3D input of shape (%d, h, w) passed (implicit single sample passed with 3D)" % (self.channels, x.shape[0])
                n_samples = 1
        elif len(x.shape) == 4:
            self.channels = x.shape[1]
            assert x.shape[1] == self.channels, "second dimension of x input must represent the image channels but dimension == %d and self.channels = %d" % (x.shape[1], self.channels)
            n_samples = x.shape[0]
        
        n = n_samples
        c = self.channels
        h = x.shape[-2]
        w = x.shape[-1]
        
        result = self.pool(x.reshape(n, c, h, w), self.levels, self.mode, self.method, n=n, c=c, h=h, w=w)
        if compress_single_dimensions:
            if n == 1 and c == 1:
                return result.reshape(self.get_output_size())
            elif n == 1:
                return result.reshape(c, self.get_output_size())
            elif c == 1:
                return result.reshape(n, self.get_output_size())
        return result


    @staticmethod
    def pool(previous_conv, levels, mode, method, n, c, h, w):
        """
        Static Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :param method defines whether "spatial" or "temporal" pooling is used

        :returns a tensor vector with shape [batch x channels x n],
                                            where n: sum(level ** p) for each level in levels where p == 1 
                                            if temporal else 2 if spatial
        """
        for i, level in enumerate(levels):
            w_kernel = level
            w_pad1 = int(math.floor((w - w) / 2))
            w_pad2 = int(math.ceil((w - w) / 2))

            if method == 'spatial':
                h_kernel = level
                h_pad1 = int(math.floor((h - h) / 2))
                h_pad2 = int(math.ceil((h - h) / 2))

                # assert w_pad1 + w_pad2 == (w_kernel * w - w) and \
                #     h_pad1 + h_pad2 == (h_kernel * h - h)

                padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                    mode='constant', value=0)
            elif method == 'temporal':
                h_kernel = h

                assert w_pad1 + w_pad2 == (w_kernel * level - w)

                padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2],
                                    mode='constant', value=0)
            
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)

            if i == 0:
                # spp = x.view(num_sample, -1)
                spp = x.reshape(n, c, -1)
            else:
                # spp = torch.cat((spp, x.view(num_sample, -1)), 1)
                spp = torch.cat((spp, x.reshape(n, c, -1)), 2)

        return spp


class SpatialPyramidPooling(PyramidPooling):
    def __init__(self, levels, channels=1, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"

                :returns (forward) a tensor vector with shape [batch x channels x n],
                                                    where n: sum(level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super(SpatialPyramidPooling, self).__init__(
            levels, channels=channels, mode=mode, method='spatial')