import torch
import torch.nn as nn

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from ..utils import PointwiseConvLayer, DWConvBlock, UpsampleLayer

@MODELS.register_module()
class StyleNet(BaseBackbone):
    def __init__(self,
                 in_channels,
                 hidden_dims = (16, 32, 64, 128),
                 expand_dims = 2,
                 img_size = (256, 192),
                 ):
        super().__init__()
        self.img_size = img_size

        self.conv0 = nn.Sequential(
            PointwiseConvLayer(in_channels, hidden_dims[0]),
            DWConvBlock(hidden_dims[0], hidden_dims[0]*expand_dims)
        )
        self.pool0 = nn.AvgPool2d(3, 2, 1)

        self.conv1 = nn.Sequential(
            PointwiseConvLayer(hidden_dims[0], hidden_dims[1]),
            DWConvBlock(hidden_dims[1], hidden_dims[1]*expand_dims)
        )
        self.pool1 = nn.AvgPool2d(3, 2, 1)

        self.conv2 = nn.Sequential(
            PointwiseConvLayer(hidden_dims[1], hidden_dims[2]),
            DWConvBlock(hidden_dims[2], hidden_dims[2]*expand_dims)
        )
        self.pool2 = nn.AvgPool2d(3, 2, 1)

        self.conv3 = nn.Sequential(
            PointwiseConvLayer(hidden_dims[2], hidden_dims[3]),
            DWConvBlock(hidden_dims[3], hidden_dims[3]*expand_dims)
        )
        self.pool3 = nn.AvgPool2d(2, 2, 0)

        self.bottom = nn.Sequential(
            DWConvBlock(hidden_dims[3], hidden_dims[3]*expand_dims)
        )

        self.up3 = UpsampleLayer(hidden_dims[3], hidden_dims[3], self.get_size(8))

        self.after3 = PointwiseConvLayer(hidden_dims[3]*2, hidden_dims[3])

        self.up2 = UpsampleLayer(hidden_dims[3], hidden_dims[2], self.get_size(4))

        self.after2 = PointwiseConvLayer(hidden_dims[2]*2, hidden_dims[2])

        self.up1 = UpsampleLayer(hidden_dims[2], hidden_dims[1], self.get_size(2))

        self.after1 = PointwiseConvLayer(hidden_dims[1]*2, hidden_dims[1])

        self.up0 = UpsampleLayer(hidden_dims[1], hidden_dims[0], self.img_size)

        self.after0 = PointwiseConvLayer(hidden_dims[0]*2, in_channels)

    def get_size(self, ratio:int):
        return (self.img_size[0]//ratio, self.img_size[1]//ratio)

    def forward(self, x:torch.Tensor):
        feat0 = self.conv0(x)

        feat1 = self.pool0(feat0)
        feat1 = self.conv1(feat1)

        feat2 = self.pool1(feat1)
        feat2 = self.conv2(feat2)

        feat3 = self.pool2(feat2)
        feat3 = self.conv3(feat3)

        feat4 = self.pool3(feat3)
        feat4 = self.bottom(feat4)

        out = torch.cat([feat3, self.up3(feat4)], dim=1)
        out = self.after3(out)

        out = torch.cat([feat2, self.up2(out)], dim=1)
        out = self.after2(out)

        out = torch.cat([feat1, self.up1(out)], dim=1)
        out = self.after1(out)

        out = torch.cat([feat0, self.up0(out)], dim=1)
        out = self.after0(out)

        out = out + x

        return (x, out)