import torch
import torch.nn as nn
import torch.nn.functional as F

class ReConv(nn.Module):
    def __init__(self,
                 in_channels = 3,
                 out_channels = 3,
                 kernel_size = 3,
                 first_groups = 1
                 ):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size, 
                               stride=1, padding=1, dilation=1, groups=first_groups)
        
        self.conv1 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size,
                               stride=1, padding=2, dilation=2, groups=out_channels)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size,
                               stride=1, padding=4, dilation=4, groups=out_channels)

    def forward(self, x:torch.Tensor):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class ReConv2(ReConv):
    def __init__(self,
                 in_channels = 3,
                 out_channels = 3,
                 kernel_size = 3,
                 first_groups = 1):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         first_groups)
        self.norm0 = nn.BatchNorm2d(out_channels)
        self.act0 = nn.SiLU()

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()

        self.norm2 = nn.BatchNorm2d(out_channels)
        pass

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.act0(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = x + out

        return out
    
class ReConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 paddings=(1, 2, 4),
                 dilations=(1, 2, 4),
                 shortcut=False):
        if (shortcut and (in_channels != out_channels)):
            ValueError(f"cant shortcut. in_channels: {in_channels}, out_channels: {out_channels}")
        super().__init__()
        self.shotcut = shortcut

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, stride,
                      padding=paddings[0], dilation=dilations[0], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1,
                      padding=paddings[1], dilation=dilations[1], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size, 1,
                      padding=paddings[2], dilation=dilations[2], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        if (self.shotcut):
            out = out + x
        return out
    
class DualReConv(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 cfg_l=dict(
                     paddings=(1, 2, 4),
                     dilations=(1, 2, 4),
                     shotcut=False
                 ),
                 cfg_r=dict(
                     paddings=(4, 2, 1),
                     dilations=(4, 2, 1),
                     shotcut=False                     
                 ),
                 shortcut=False):
        super().__init__()
        if (shortcut and (in_channels != out_channels)):
            ValueError(f"cant shortcut. in_channels: {in_channels}, out_channels: {out_channels}")
        self.shortcut = shortcut

        self.pw_front = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())

        self.reconv_l = ReConvModule(
            hidden_channels, hidden_channels, hidden_channels, kernel_size, stride,
            paddings=cfg_l["paddings"], dilations=cfg_l["dilations"], shortcut=cfg_l["shotcut"]
        )

        self.reconv_r = ReConvModule(
            hidden_channels, hidden_channels, hidden_channels, kernel_size, stride,
            paddings=cfg_r["paddings"], dilations=cfg_r["dilations"], shortcut=cfg_r["shotcut"]
        )

        self.pw_back = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        out = self.pw_front(x)
        out = self.reconv_l(out) + self.reconv_r(out)
        out = self.pw_back(out)
        if (self.shortcut):
            out = out + x
        return out
    
class ReConvNeXt(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 paddings = (1, 2, 4),
                 dilations = (1, 2, 4),
                 shortcut = False
                 ):
        super().__init__()
        if (shortcut and (in_channels != out_channels)):
            ValueError(f"cant shortcut. in_channels: {in_channels}, out_channels: {out_channels}")

        self.shortcut = shortcut

        self.pw_conv_front = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())

        self.conv0 = nn.Sequential(
            nn.Conv2d(hidden_channels, 
                    hidden_channels, 
                    kernel_size, 
                    stride=stride, padding=paddings[0], dilation=dilations[0], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size,
                      stride=stride, padding=paddings[1], dilation=dilations[1], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size,
                      stride=stride, padding=paddings[2], dilation=dilations[2], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())
                
        self.pw_conv_back = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x:torch.Tensor):
        out = self.pw_conv_front(x)
        out = self.conv0(out) + self.conv1(out) + self.conv2(out)
        out = self.pw_conv_back(out)
        if self.shortcut:
            out = out + x
        return out

class ReConvNeXt2(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 paddings = (1, 2, 4),
                 dilations = (1, 2, 4),
                 shortcut = False
                 ):
        super().__init__()
        if (shortcut and (in_channels != out_channels)):
            ValueError(f"cant shortcut. in_channels: {in_channels}, out_channels: {out_channels}")

        self.shortcut = shortcut

        self.pw_conv_front = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())

        self.conv0 = nn.Sequential(
            nn.Conv2d(hidden_channels, 
                    hidden_channels, 
                    kernel_size, 
                    stride=stride, padding=paddings[0], dilation=dilations[0], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size,
                      stride=stride, padding=paddings[1], dilation=dilations[1], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size,
                      stride=stride, padding=paddings[2], dilation=dilations[2], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())
                
        self.pw_conv_back = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU())

    def forward(self, x:torch.Tensor):
        out = self.pw_conv_front(x)
        out = self.conv0(out) + self.conv1(out) + self.conv2(out)
        out = self.pw_conv_back(out)
        if self.shortcut:
            out = out + x
        return out

class ReConvNeXt3(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size = 3,
                 stride = 1,
                 paddings = (1, 2, 4),
                 dilations = (1, 2, 4),
                 shortcut = False
                 ):
        super().__init__()
        if (shortcut and (in_channels != out_channels)):
            ValueError(f"cant shortcut. in_channels: {in_channels}, out_channels: {out_channels}")

        self.shortcut = shortcut

        self.pw_conv_front = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())
            
        self.conv0 = nn.Sequential(
            nn.Conv2d(hidden_channels, 
                    hidden_channels, 
                    kernel_size, 
                    stride=stride, padding=paddings[0], dilation=dilations[0], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size,
                      stride=1, padding=paddings[1], dilation=dilations[1], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size,
                      stride=1, padding=paddings[2], dilation=dilations[2], groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU())

        self.pw_conv_back = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU())

    def forward(self, x):
        out = self.pw_conv_front(x)

        out = self.conv0(out)
        out1 = self.conv1(out)
        out = out + out1 + self.conv2(out1)

        out = self.pw_conv_back(out)

        if self.shortcut:
            out = out + x

        return out