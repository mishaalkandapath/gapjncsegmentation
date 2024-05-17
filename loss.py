import os
import cv2
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchio as tio

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=3, device=torch.device("cpu")):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = gamma # larger gamma values focus more on hard examples
        self.alpha = torch.tensor(alpha).to(device)
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (batch_size, class=2, depth, height, width)
        pt = torch.exp(-bce_loss)
        targets = targets.to(torch.int64) # convert to int64 for indexing
        focal_loss = self.alpha[targets] * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=0.75, device=torch.device("cpu")):
        super(FocalTverskyLoss, self).__init__()
        self.device = device
        self.gamma = gamma
        self.alpha = torch.Tensor([alpha]).to(device)
        self.beta = torch.Tensor([beta]).to(device)
    
    def forward(self, inputs, targets, smooth=1):
        true_pos = torch.sum(targets * inputs, dim=(1,2,3,4))
        false_neg = torch.sum(targets * (1-inputs), dim=(1,2,3,4))
        false_pos = torch.sum((1-targets) * inputs, dim=(1,2,3,4))
        print(f"True pos: {true_pos}, False neg: {false_neg}, False pos: {false_pos}")
        tversky = (true_pos + smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + smooth)
        tversky_loss = 1 - tversky
        focal_tversky_loss = torch.pow(tversky_loss, self.gamma)
        return focal_tversky_loss.mean()

class FocalTverskyLossWith2d3d(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=0.75, device=torch.device("cpu"), intermediate_weight = 0.33):
        super(FocalTverskyLoss, self).__init__()
        self.intermediate_loss = FocalTverskyLoss(alpha, beta, gamma, device)
        self.final_loss = FocalTverskyLoss(alpha, beta, gamma, device)
    
    def forward(self, preds_2d, preds_3d, targets):
        loss_2d = self.intermediate_loss(preds_2d, targets) # intermediate 2D class predictions loss
        loss_3d = self.final_loss(preds_3d, targets) # final 3D class predictions
        return loss_3d + self.c_2d * loss_2d
    

class FocalLossWith2d3d(nn.Module):
    def __init__(self, alpha, gamma=2, device=torch.device("cpu")):
        """ 
        Args:
            alpha: tensor of shape (2,) with the alpha values for the focal loss
        """
        super(FocalLossWith2d3d, self).__init__()
        self.c_2d = 0.33 # constant that weighs importance of intermediate 2D class predictions in the loss function
        self.intermediate_loss = FocalLoss(alpha, gamma, device)
        self.final_loss = FocalLoss(alpha, gamma, device)
    
    def forward(self, preds_2d, preds_3d, targets):
        loss_2d = self.intermediate_loss(preds_2d, targets) # intermediate 2D class predictions loss
        loss_3d = self.final_loss(preds_3d, targets)
        return loss_3d + self.c_2d * loss_2d


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
        