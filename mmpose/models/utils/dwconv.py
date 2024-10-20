import torch
from torch import nn

class PointwiseConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inplace=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # self.norm = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-6)
        self.act = nn.SiLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DWConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 inplace=False
                 ):
        super().__init__()
        self.pw_conv0 = PointwiseConvLayer(in_channels, hidden_channels)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 
                      hidden_channels, 
                      kernel_size, 
                      stride, 
                      padding, 
                      groups=hidden_channels),
            # nn.BatchNorm2d(hidden_channels, eps=1e-6)
            nn.InstanceNorm2d(hidden_channels, eps=1e-6)
        )
        self.act0 = nn.SiLU(inplace=inplace)

        self.pw_conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1),
            # nn.BatchNorm2d(in_channels, eps=1e-6)
            nn.InstanceNorm2d(in_channels, eps=1e-6)
        )
        self.act1 = nn.SiLU(inplace=inplace)

    def forward(self, x:torch.Tensor):
        feat = self.pw_conv0(x)
        feat = feat + self.dw_conv(feat)
        feat = self.act0(feat)
        feat = self.pw_conv1(feat)
        x = x + feat
        x = self.act1(x)
        return x