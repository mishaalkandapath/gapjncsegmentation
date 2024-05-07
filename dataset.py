import numpy as np
import cv2 
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import v2
import re
import random

class SectionsDataset(Dataset):
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            image_dim = (512, 512),
    ):    
        
        self.image_path_triples = []
        self.mask_path_triples = []
        s_repl = lambda s: "s00" + str(s)+"_" if s <= 9 else "s0"+str(s)+"_"

        # get all the image ids
        for img_id in sorted(os.listdir(images_dir)):
            dept_str = re.findall(r's\d*_',img_id)[0]
            cur_depth = int(dept_str[1:-1])
            nxt, nnxt = img_id.replace(dept_str, s_repl(cur_depth+1)), img_id.replace(dept_str, s_repl(cur_depth+2))
            if os.path.isfile(os.path.join(images_dir, nxt)) and os.path.isfile(os.path.join(images_dir, nnxt)):
                self.image_path_triples.append(os.path.join(images_dir, img_id), os.path.join(images_dir, nxt), os.path.join(images_dir, nnxt))
                self.mask_path_triples.append(os.path.join(masks_dir, img_id),os.path.join(masks_dir, nxt), os.path.join(masks_dir, nnxt))
        
        self.image_paths = ([os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))])
        self.mask_paths = ([os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))])
        self.augmentation = augmentation 
        self.image_dim = image_dim

        self.transform1 = v2.Compose([
            v2.RandomRotation(degrees=(0, 180)), 
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.0))
        ])
        self.transform2 = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=1),
                v2.RandomVerticalFlip(p=1),
               v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.0))
            ]
        )
        self.transform3 = v2.Compose(
            [
                 v2.RandomRotation(degrees=(0, 180)), 
                v2.RandomHorizontalFlip(p=1),
                v2.RandomVerticalFlip(p=1),
               v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.0))
            ]
        )
    
    def __getitem__(self, i):
        images = self.image_path_triples[i]
        masks = self.mask_path_triples[i]

        img, m = [], []
        for j in range(3):
            # read images and masks # they have 3 values (BGR) --> read as 2 channel grayscale (H, W)
            image = cv2.cvtColor(cv2.imread(images[j]), cv2.COLOR_BGR2GRAY) # each pixel is 
            mask = cv2.cvtColor(cv2.imread(masks[j]), cv2.COLOR_BGR2GRAY) # each pixel is 0 or 255
            img.append(image)
            m.append(mask)
        
        image, mask = np.stack(img, axis=0), np.stack(m, axis=0)

        # make sure each pixel is 0 or 255
        mask_labels = np.unique(mask)
        if (len(mask_labels)>2):
            print("More than 2 labels found for mask")
        num_one = mask_labels[0]
        mask[mask != num_one] = 0
        mask[mask == num_one] = 1
        
        
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
        _transform.append(v2.ToTensor())

        img_size = 512
        width, height = self.image_dim
        max_dim = max(img_size, width, height)
        pad_left = (max_dim-width)//2
        pad_right = max_dim-width-pad_left
        pad_top = (max_dim-height)//2
        pad_bottom = max_dim-height-pad_top
        _transform.append(v2.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), 
                                        padding_mode='edge'))
        _transform.append(v2.Resize(interpolation=v2.InterpolationMode.NEAREST_EXACT,size=(img_size, img_size)))  
        mask = v2.Compose(_transform)(mask)

        # ont_hot_mask = F.one_hot(mask.long(), num_classes=2).squeeze().permute(2, 0, 1).float()
        ont_hot_mask = mask

        _transform_img = _transform.copy()
        _transform_img.append(v2.Normalize(mean=[0.5], std=[0.5]))
        image = v2.Compose(_transform_img)(image)
            
        return image, ont_hot_mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)
