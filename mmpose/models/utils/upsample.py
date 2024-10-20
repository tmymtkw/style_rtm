import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleLayer(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 feat_size:tuple[int, int]):
        super().__init__()
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x:torch.Tensor):
        feat = self.pw_conv(x)
        feat = F.interpolate(feat, scale_factor=2., mode="bilinear")
        feat = self.norm(feat)
        feat = self.act(feat)
        return feat