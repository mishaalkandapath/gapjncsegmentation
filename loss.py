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

