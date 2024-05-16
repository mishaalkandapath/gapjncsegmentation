import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# --- correspond to arrows in paper figure ---
class DoubleConv(nn.Module):
    """Two Conv2D + BN + ReLU
    
    Note: "Conv2D" is a convolutional layer with kernel size 3x3, or 1x3x3 in 3D
    
    Attributes:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 1x3x3 conv + BN + ReLU
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # 1x3x3 conv + BN + ReLU
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): 5D input tensor of shape (batch_size, channel=1, depth, height, width)
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
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

# --- group into blocks for easier reading ---
class DownBlock(nn.Module):
    """Double Convolution followed by Max Pooling
    
    Attributes:
        add_3d (bool): whether to add 3D convolution before double conv
    """
    def __init__(self, in_channels, out_channels, add_3d = False):
        super(DownBlock, self).__init__()
        self.add_3d = add_3d
        self.in_channels = in_channels
        self.out_channels = out_channels
        if add_3d:
            # add 3d conv before double conv
            self.conv_3d = nn.Conv3d(in_channels, out_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
            self.double_conv = DoubleConv(out_channels//2, out_channels)
        else:
            self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool3d((1,2,2), stride=(1,2,2))

    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): 5D input tensor of shape (batch_size, channel=1, depth, height, width)
        """
        # apply 3D conv if needed
        if self.add_3d:
            x = self.conv_3d(x) # x: (1, 32, 5, 512, 512)
        skip_out = self.double_conv(x) # (1, 64, 5, 512, 512)
        down_out = self.down_sample(skip_out) # (1, 64, 5, 256, 256)
        return down_out, skip_out

class UpBlock(nn.Module):
    """Up Convolution (Upsampling followed by Double Convolution)"""
    def __init__(self, in_channels, out_channels, add_3d = False):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_3d = add_3d
        self.up_sample = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=(1,2,2), stride=(1,2,2)) # doubles number of channels
        # same number of channels
        if add_3d:
            self.conv3d = nn.Conv3d(in_channels, out_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
            self.double_conv = DoubleConv(out_channels//2, out_channels)
        else:
            self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        """ 
        Args:
            down_input (torch.Tensor): 5D input tensor of shape (batch_size, in_channels, depth, height, width)
            skip_input (torch.Tensor): 5D input tensor from skip connection (batch_size, in_channels//2, depth, height*2, width*2)
        """
        x = self.up_sample(down_input) # -> (batch_size, in_channels//2, depth, height*2, width*2) # turn down_input into same shape as skip_input
        x = torch.cat([x, skip_input], dim=1) # -> (batch_size, in_channels, depth, height*2, width*2) # concatenate along channels
        if self.add_3d:
            x = self.conv3d(x)
        x = self.double_conv(x)
        return x
        
class PyramidBlock(nn.Module):
    """ 3D Spatial Pyramid Pooling Block (3x3x3 conv + BN + ReLU)"""
    def __init__(self, num_convs, num_channels):
        super(PyramidBlock, self).__init__()
        self.num_convs = num_convs
        self.num_channels = num_channels
        self.pyramid = nn.Sequential()
        for i in range(num_convs):
            self.pyramid.add_module(f"conv{i}", nn.Conv3d(num_channels, num_channels, kernel_size=(3,3,3), padding=1))
            self.pyramid.add_module(f"bn{i}", nn.BatchNorm3d(num_channels))
            self.pyramid.add_module(f"relu{i}", nn.ReLU(inplace=True))
            
    
    def forward(self, x):
        x = self.pyramid(x)
        # for i in range(self.num_convs):
        #     x = nn.Conv3d(self.num_channels, self.num_channels, kernel_size=(3,3,3), padding=1)(x)
        #     x = nn.BatchNorm3d(self.num_channels)(x)
        #     x = nn.ReLU(inplace=True)(x)
        return x
    
# --- put it all together ---
class UNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2):
        """Initialize the UNet model"""
        super(UNet, self).__init__()
        
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, add_3d=True) # 1 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, add_3d=True) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        
        # Bottleneck
        self.double_conv = DoubleConv(256, 512)
        
        # Upsampling Path
        self.up_conv3 = UpBlock(512, 256)
        self.up_conv2 = UpBlock(256, 128, add_3d=True)
        self.up_conv1 = UpBlock(128, 64, add_3d=True)
        
        # 1x1x1 conv + ReLU
        self.one_conv1 = SingleConv1D(64, out_classes)
        self.softmax = nn.Softmax(dim=1)
        
        # 3D Spatial Pyramid
        # -- side 1
        self.single_three_conv1 = SingleConv3D(64+out_classes, 64)
        self.pyramid1 = PyramidBlock(num_convs=2, num_channels=64)
        self.res_conn = nn.Conv3d(64, 32, kernel_size=(1,1,1))
        
        # -- side 2
        self.single_three_conv2 = SingleConv3D(64+out_classes, 128)
        self.pyramid2 = PyramidBlock(num_convs=7, num_channels=128)
        self.single_conv2 = SingleConv3D(128, 64)
        self.transpose = nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2))
        
        # Final Convolution
        self.single_conv3 = SingleConv1D(64, out_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """Forward pass of the UNet model
        Args:
            x (torch.Tensor): 5D input tensor (batch_size, channel=1, depth, height, width)
                - note: in pytorch docs, (N, C, D, H, W) is used
        """
        # Encoder-Decoder Path (depth doesn't change)
        x, skip1_out = self.down_conv1(x) # x: (1, 64, 5, 150, 150)
        x, skip2_out = self.down_conv2(x) # x: (1, 128, 5, 75, 75)
        x, skip3_out = self.down_conv3(x) # x: (1, 256, 5, 37, 37)
        x = self.double_conv(x) # x: (1, 512, 5, 37, 37)
        x = self.up_conv3(x, skip3_out) # x: (1, 256, 5, 75, 75)
        x = self.up_conv2(x, skip2_out) # x: (1, 128, 5, 150, 150)
        out1 = self.up_conv1(x, skip1_out) # x: (1, 64, 5, 300, 300)
        
        x = self.one_conv1(out1) # x: (1, 2, 5, 300, 300)
        intermediate = self.softmax(x) # x: (1, 2, 5, 300, 300)
        x = torch.cat([intermediate, out1], dim=1) # x: (5, 66, 300, 300)
        
        # 3D Spatial Pyramid
        # - side 1
        x1 = self.single_three_conv1(x) # x: (1, 66, 5, 300, 300)
        x1 = self.pyramid1(x1) # x: (1, 64, 5, 300, 300)
        x1 = self.res_conn(x1)
        
        x2 = nn.MaxPool3d((1,2,2), stride=(1,2,2))(x) # x: (5, 66, 150, 150)
        x2 = self.single_three_conv2(x2)
        x2 = self.pyramid2(x2)
        x2 = self.single_conv2(x2)
        x2 = self.transpose(x2)
        
        # Final Convolution
        final = torch.cat([x1, x2], dim=1) # x: (5, 64, 150, 150)
        final = self.single_conv3(final) # x: (5, 2, 150, 150)
        final = self.softmax(final) # x: (5, 2, 150, 150)
        
        return intermediate, final

