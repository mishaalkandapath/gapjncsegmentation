import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# correspond to arrows in paper figure
class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # 1x3x3 conv + BN + ReLU
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # 1x3x3 conv + BN + ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class SingleConv3D(nn.Module):
    """ 3x3x3 Convolution followed by BN and ReLU"""
    def __init__(self, in_channels, out_channels):
        super(SingleConv3D, self).__init__()
        self.single_conv = nn.Sequential(
            # 3x3x3 conv + BN + ReLU
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)
    
class SingleConv1D(nn.Module):
    """ 1x1x1 Convolution followed by BN and ReLU"""
    def __init__(self, in_channels, out_channels):
        super(SingleConv1D, self).__init__()
        self.single_conv = nn.Sequential(
            # 1x1x1 conv + BN + ReLU
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

# group into blocks for easier reading
class DownBlock(nn.Module):
    """Double Convolution followed by Max Pooling"""
    def __init__(self, in_channels, out_channels, add_3d = False):
        super(DownBlock, self).__init__()
        if add_3d:
            # add 3d conv before double conv
            self.conv_3d = nn.Conv3d(in_channels, out_channels//2, kernel_size=(3,3,3), padding=1)
            self.double_conv = DoubleConv(out_channels//2, out_channels)
        else:
            self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    """Up Convolution (Upsampling followed by Double Convolution)"""
    def __init__(self, in_channels, out_channels, up_sample_mode='conv_transpose', add_3d = False):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        
        if add_3d:
            self.conv3d = nn.Conv3d(in_channels, out_channels//2, kernel_size=(3,3,3), padding=1)
            self.double_conv = DoubleConv(out_channels//2, out_channels)
        else:
            self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)
    
class PyramidBlock(nn.Module):
    """ 3D Spatial Pyramid Pooling Block (3x3x3 conv + BN + ReLU)"""
    def __init__(self, num_convs, num_channels):
        super(PyramidBlock, self).__init__()
    
    def forward(self, x):
        for i in range(self.num_convs):
            x = nn.Conv3d(self.num_channels, self.num_channels, kernel_size=(3,3,3), padding=1)(x)
            x = nn.BatchNorm3d(self.num_channels)(x)
            x = nn.ReLU(inplace=True)(x)
        return x

# model architecture
class UNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        """Initialize the UNet model"""
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, add_3d=True) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, add_3d=True) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        
        # Bottleneck
        self.double_conv = DoubleConv(256, 512)
        
        # Upsampling Path
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode) # 256 + 512 input channels --> 256 output channels
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode, add_3d=True)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode, add_3d=True)
        
        # 1x1x1 conv + ReLU
        self.one_conv1 = SingleConv1D(64, out_classes)
        self.softmax = nn.Softmax(dim=1)
        
        # 3D Spatial Pyramid
        # -- side 1
        self.single_three_conv1 = SingleConv3D(64+out_classes, 64)
        self.pyramid1 = PyramidBlock(num_convs=2, num_channels=64)
        
        # -- side 2
        self.single_three_conv2 = SingleConv3D(out_classes, 128)
        self.pyramid2 = PyramidBlock(num_convs=7, num_channels=128)
        self.transpose = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Final Convolution
        self.single_conv2 = SingleConv1D(128, out_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """Forward pass of the UNet model
        Args:
            x (torch.Tensor): input tensor (depth, channels=1, height, width)
        """
        # Encoder-Decoder Path (depth doesn't change)
        x, skip1_out = self.down_conv1(x) # x: (5, 64, 150, 150), skip1_out: (16, 64, 512, 512) (depth, channels, height, width)
        x, skip2_out = self.down_conv2(x) # x: (5, 128, 75, 75)
        x, skip3_out = self.down_conv3(x) # x: (5, 256, 37, 37)
        x = self.double_conv(x) # x: (5, 512, 75, 75)
        x = self.up_conv3(x, skip3_out) # x: (5, 256, 150, 150)
        x = self.up_conv2(x, skip2_out) # x: (5, 128, 300, 300)
        out1 = self.up_conv1(x, skip1_out) # x: (5, 64, 300, 300)        
        
        x = self.one_conv1(out1) # skip_out: (5, 2, 300, 300)
        x = self.softmax(x)
        x = torch.cat([x, out1], dim=1) # x: (5, 66, 300, 300)
        
        # 3D Spatial Pyramid
        x1 = self.single_three_conv1(x)
        x1 = self.pyramid1(x1)
        
        x2 = nn.MaxPool2d(2, stride=2)(x) # x: (5, 66, 150, 150)
        x2 = self.single_three_conv2(x2)
        x2 = self.pyramid2(x2)
        x2 = self.transpose1(x2)
        
        # Final Convolution
        x = torch.cat([x1, x2], dim=1) # x: (5, 64, 150, 150)
        x = self.single_conv1(x)
        x = self.softmax(x) # x: (5, 2, 150, 150)
        
        return x

