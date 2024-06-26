"""
loss.py: Contains loss functions for training the UNet model, 
and functions to calculate weights for class imbalance in the dataset
"""

import os
import cv2
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchio as tio

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=3):
        super(FocalLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = torch.tensor(gamma).to(device)
        self.alpha = torch.tensor(alpha).to(device)
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (batch_size, class=2, depth, height, width)
        pt = torch.exp(-bce_loss)
        targets = targets.to(torch.int64) # convert to int64 for indexing
        focal_loss = self.alpha[targets] * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.gamma = torch.tensor(gamma).to(device)

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky

class FocalTverskyLossWith2d3d(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=0.75, intermediate_weight = 0.33):
        super(FocalTverskyLossWith2d3d, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intermediate_weight = torch.Tensor([intermediate_weight]).to(device) # constant that weighs importance of intermediate 2D class predictions in the loss function
        self.intermediate_loss = FocalTverskyLoss(alpha, beta, gamma, device)
        self.final_loss = FocalTverskyLoss(alpha, beta, gamma, device)
    def forward(self, preds_2d, preds_3d, targets):
        loss_2d = self.intermediate_loss(preds_2d, targets) # intermediate 2D class predictions loss
        loss_3d = self.final_loss(preds_3d, targets) # final 3D class predictions
        return loss_3d + self.intermediate_weight * loss_2d
    
class FocalLossWith2d3d(nn.Module):
    def __init__(self, alpha, gamma=2, intermediate_weight=0.33):
        """ 
        Args:
            alpha: tensor of shape (2,) with the alpha values for the focal loss
        """
        super(FocalLossWith2d3d, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intermediate_weight = torch.Tensor([intermediate_weight]).to(device) # constant that weighs importance of intermediate 2D class predictions in the loss function
        self.intermediate_loss = FocalLoss(alpha, gamma, device)
        self.final_loss = FocalLoss(alpha, gamma, device)
    
    def forward(self, preds_2d, preds_3d, targets):
        loss_2d = self.intermediate_loss(preds_2d, targets) # intermediate 2D class predictions loss
        loss_3d = self.final_loss(preds_3d, targets)
        return loss_3d + self.intermediate_weight * loss_2d

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class ComboLoss(nn.Module):
    """ 
    Attributes:
    alpha:  < 0.5 penalises FP more, > 0.5 penalises FN more
    ce_ratio: weighted contribution of modified CE loss compared to Dice loss
    """
    def __init__(self, alpha, ce_ratio, weight=None, size_average=True):
        super(ComboLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = torch.tensor(alpha).to(device)
        self.ce_ratio = torch.tensor(ce_ratio).to(device)

    def forward(self, inputs, targets, smooth=1, eps=1e-7):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        positive_loss = self.alpha * (targets * torch.log(inputs))
        negative_loss = (1 - self.alpha) * ((1.0 - targets) * torch.log(1.0 - inputs))
        out = - (positive_loss + negative_loss)
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) + ((1 - self.ce_ratio) * dice)
        return combo

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def calculate_alpha(train_dataset):
    """ 
    Calculate the alpha values for the focal loss function.
    The alpha values are inversely proportional to the class frequencies in the dataset.
    
    Ex. 
    class 0 has 90 samples and class 1 has 10 samples
    class frequencies are 90/100 and 10/100
    inverse class frequencies are 100/90 and 100/10
    alpha is proportional to 100/90 and 100/10, so it can be set as [10/90, 90/90] or [0.11, 1]
    
    Args:
    train_dataset (torch.utils.data.Dataset): the training dataset
    """
    smushed_labels = None
    for i in range(len(train_dataset)):
        if i == 0:
            depth, height, width = train_dataset[i][0].shape
        if smushed_labels is None: smushed_labels = train_dataset[i][1].to(torch.int64)
        else: smushed_labels = torch.concat([smushed_labels, train_dataset[i][1].to(torch.int64)])
        print(f"Processed {i+1}/{len(train_dataset)} images", end="\r")
    class_counts = torch.bincount(smushed_labels.flatten())
    total_samples = len(train_dataset) * depth * height * width
    
    w1, w2 = 1/(class_counts[0]/total_samples), 1/(class_counts[1]/total_samples)
    cls_weights = torch.Tensor([w1, w2/9])
    return cls_weights

def get_inverse_class_frequencies(train_dataset, num_classes=2):
    """ 
    Calculate the inverse class frequencies
    
    Args:
    train_dataset (torch.utils.data.Dataset): the training dataset
    """
    
    total_num_zeros = 0
    for i in range(len(train_dataset)):
        inputs, labels = train_dataset[i]
        if i == 0:
            _, depth, height, width = inputs.shape
        total_num_zeros += torch.sum(labels[1])
        print(f"Processed {i+1}/{len(train_dataset)} images", end="\r")
    
    total_pixels_img = depth * height * width
    total_pixels = total_pixels_img * len(train_dataset)
    total_num_ones = total_pixels - total_num_zeros
    
    class_frequencies = [(num_classes * total_num_zeros) / total_pixels, (num_classes * total_num_ones) / total_pixels]
    inverse_class_frequencies = [1/freq for freq in class_frequencies]
    return inverse_class_frequencies

def get_class_frequencies(train_dataset, num_classes=2):
    """ 
    Calculate the inverse class frequencies

    negative class: 0
    positive class: 1
    
    Args:
    train_dataset (torch.utils.data.Dataset): the training dataset
    """
    total_negatives = 0
    total_positives = 0
    for i in range(len(train_dataset)):
        inputs, labels = train_dataset[i]
        if i == 0:
            _, depth, height, width = inputs.shape
        total_num_zeros += torch.sum(labels[1])
        print(f"Processed {i+1}/{len(train_dataset)} images", end="\r")
    
    total_pixels_img = depth * height * width
    total_pixels = total_pixels_img * len(train_dataset)
    total_positives = total_pixels - total_negatives
    
    class_frequencies = [total_negatives / total_pixels, total_positives / total_pixels]
    return class_frequencies

def scale_to_sum_to_one(alpha):
    """ 
    Scale the alpha values to sum to one
    
    Args:
    alpha (torch.Tensor): the alpha values
    """
    return alpha / torch.sum(alpha)
        