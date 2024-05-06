import numpy as np
import cv2 
import os
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch 


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DownBlock(nn.Module):
    """Double Convolution followed by Max Pooling"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    """Up Convolution (Upsampling followed by Double Convolution)"""
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class SetNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, num_images=3, up_sample_mode='conv_transpose'):
        """Initialize the UNet model"""
        super(SetNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, self.num_images, kernel_size=1) 
        
        self.num_images = num_images

    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 3, 512, 512)
        """
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width)
        x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        x, skip3_out = self.down_conv3(x) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 3, 512, 512)
        return x