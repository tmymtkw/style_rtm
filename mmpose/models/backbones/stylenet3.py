import torch
import torch.nn as nn

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from ..utils import PointwiseConvLayer, DWConvBlock, UpsampleLayer, PatchEmbedLayer, NormalEncoderBlock, DecoderBlock
from .uvit import UpLayer

class StyleEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dims = (12, 24, 48, 128),
                 expand_dims = 1.5,
                 img_size = (256, 192),
                 drop_rate=0.1,
                 cls_num=20,
                 reduction=16,
                 inplace=False
                 ):
        super().__init__()
        self.img_size = img_size

        self.conv0 = nn.Sequential(
            PointwiseConvLayer(in_channels, hidden_dims[0], inplace=inplace),
            DWConvBlock(hidden_dims[0], int(hidden_dims[0]*expand_dims), inplace=inplace),
        )
        self.pool0 = nn.MaxPool2d(3, 2, 1)

        self.conv1 = nn.Sequential(
            PointwiseConvLayer(hidden_dims[0], hidden_dims[1], inplace=inplace),
            DWConvBlock(hidden_dims[1], int(hidden_dims[1]*expand_dims), inplace=inplace),
        )
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.conv2 = nn.Sequential(
            PointwiseConvLayer(hidden_dims[1], hidden_dims[2], inplace=inplace),
            DWConvBlock(hidden_dims[2], int(hidden_dims[2]*expand_dims), inplace=inplace),
        )
        self.pool2 = nn.MaxPool2d(3, 2, 1)

        self.conv3 = nn.Sequential(
            PointwiseConvLayer(hidden_dims[2], hidden_dims[3]),
            DWConvBlock(hidden_dims[3], int(hidden_dims[3]*expand_dims))
        )
        self.pool3 = nn.MaxPool2d(2, 2, 0)

        self.bottom = nn.Sequential(
            DWConvBlock(hidden_dims[3], int(hidden_dims[3]*expand_dims), inplace=inplace),
            nn.Conv2d(hidden_dims[3], hidden_dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[3]),
            nn.SiLU(inplace=inplace)
        )

        self.drop = nn.Dropout(drop_rate, inplace=inplace)
        self.mlp = nn.Sequential(
            nn.Linear(self.get_size()*hidden_dims[3], self.get_size()*hidden_dims[3]//reduction),
            nn.ReLU(),
            nn.Linear(self.get_size()*hidden_dims[3]//reduction, cls_num)
        )
        # self.mlp = nn.Linear(self.get_size()*hidden_dims[3], cls_num)

    def get_size(self):
        return self.img_size[0] * self.img_size[1] // (32 * 32)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.bottom(x)
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        x = self.mlp(x)

        return x

@MODELS.register_module()
class StyleNet3(BaseBackbone):
    def __init__(self,
                 in_channels,
                 emb_dims = 96,
                 patch_size = 8,
                 img_size = (256, 192),
                 out_size = (64, 48),
                 hidden_dims = 96*4,
                 num_blocks = (3, 6, 3),
                 dec_block = 3,
                 heads = (6, 12),
                 drop_rate = 0.1,
                 has_token = True
                 ):
        super().__init__()
        # var
        self.emb_dims = emb_dims
        self.out_dim = emb_dims
        self.img_size = img_size
        self.out_size = (out_size[0]//2, out_size[1]//2)
        # module
        self.emb = PatchEmbedLayer(in_channels, emb_dims, patch_size, img_size, has_token)

        self.m_front = nn.Sequential()
        for _ in range(num_blocks[0]):
            self.m_front.append(NormalEncoderBlock(emb_dims, heads[0], hidden_dims, drop_rate, inplace=False))

        self.down = nn.Sequential(
            nn.Conv2d(emb_dims, emb_dims, kernel_size=2, stride=2),
            nn.LayerNorm([emb_dims, self.out_size[0]//2, self.out_size[1]//2]),
            nn.GELU()
        )

        self.bot_dec = nn.ModuleList()
        for _ in range(dec_block):
            self.bot_dec.append(DecoderBlock(emb_dims, 1, hidden_dims, drop_rate))

        self.bot = nn.Sequential()
        for _ in range(num_blocks[1]):
            self.bot.append(NormalEncoderBlock(emb_dims, heads[1], hidden_dims, drop_rate, inplace=False))

        self.up = UpLayer(emb_dims, emb_dims, self.out_size)

        self.m_behind = nn.Sequential()
        for _ in range(num_blocks[2]):
            self.m_behind.append(NormalEncoderBlock(emb_dims, heads[0], hidden_dims, drop_rate, inplace=False))

        self.final = nn.Sequential(
            nn.ConvTranspose2d(emb_dims, self.out_dim, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([self.out_dim, self.out_size[0]*2, self.out_size[1]*2]),
            nn.GELU()
        )

        self.style_encoder = StyleEncoder(in_channels, drop_rate=drop_rate, inplace=False)
        self.style_emb = nn.Linear(20, emb_dims)

    def get_size(self, ratio:int):
        return (self.img_size[0]//ratio, self.img_size[1]//ratio)

    def forward(self, x:torch.Tensor):
        B = x.shape[0]

        feat_style = self.style_encoder(x)
        feat_cls = self.style_emb(feat_style)
        feat_cls = feat_cls.unsqueeze(1)

        feat = self.emb(x)
        feat = self.m_front(feat)
        token = feat[:, :1]
        feat = feat[:, 1:]

        feat_s = self.reshape_feat(feat, B, 1)
        feat_s = self.down(feat_s)
        feat_s = self.flat_feat(feat_s)
        feat_s = torch.cat([token, feat_s], dim=1)
        feat_s = self.bot(feat_s)
        for layer in self.bot_dec:
            feat_s = layer(feat_s, feat_cls)
        feat_s = feat_s[:, 1:]
        feat_s = self.reshape_feat(feat_s, B, 2)
        feat_s = self.up(feat_s)
        feat_s = self.flat_feat(feat_s)

        feat = feat + feat_s
        feat = self.m_behind(feat)

        feat = feat.reshape((x.shape[0], self.out_size[0], self.out_size[1], self.emb_dims)).permute(0, 3, 1, 2)
        feat = self.final(feat)

        return tuple([feat_style, feat])
    
    def reshape_feat(self, feat:torch.Tensor, b, factor=1):
        return feat.reshape((b, self.out_size[0]//factor, self.out_size[1]//factor, self.emb_dims)).permute(0, 3, 1, 2)

    def flat_feat(self, feat:torch.Tensor):
        feat = feat.flatten(2)
        feat = feat.transpose(1, 2)
        return feat