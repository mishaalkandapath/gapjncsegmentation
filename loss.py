import os
import cv2
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, device=torch.device("cpu")):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.device = device
        self.alpha = alpha.to(device)
    
    def forward(self, inputs, targets, width=512, height=512):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (batch_size, class, depth, height, width)
        pt = torch.exp(-bce_loss) # take the exp of the negative bce loss to get the probability of the correct class
        targets = targets.to(torch.int64) # convert to int64 for indexing
        loss = self.alpha[targets] * (1-pt)**self.gamma * bce_loss
        return loss.mean() 


def calculate_alpha(train_dataset):
    """ 
    Calculate the alpha values for the focal loss function.
    The alpha values are inversely proportional to the class frequencies in the dataset.
    
    Args:
    train_dataset (torch.utils.data.Dataset): the training dataset
    """
    smushed_labels = None
    for i in range(len(train_dataset)):
        if smushed_labels is None: smushed_labels = train_dataset[i][1].to(torch.int64)
        else: smushed_labels = torch.concat([smushed_labels, train_dataset[i][1].to(torch.int64)])
        print(f"Processed {i+1}/{len(train_dataset)} images", end="\r")
    class_counts = torch.bincount(smushed_labels.flatten())
    total_samples = len(train_dataset) * 512 * 512
    w1, w2 = 1/(class_counts[0]/total_samples), 1/(class_counts[1]/total_samples)
    cls_weights = torch.Tensor([w1, w2/9])
    print(cls_weights)