import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .myvit import PatchEmbedLayer, NormalEncoderBlock
from .uvit import UpLayer

@MODELS.register_module()
class UViT4(BaseBackbone):
    def __init__(self,
                 in_channels = 3,
                 emb_dims = (24, 96, 384),
                 hidden_dims = (24*4, 96*4, 384*4),
                 patch_size = 4,
                 img_size = (256, 192),
                 heads = (3, 6, 12),
                 num_blocks = (2, 2, 3, 2, 2),
                 drop_rate = 0.1,
                 has_token = False):
        super().__init__()
        # var
        self.emb_dims = emb_dims
        self.out_size0 = self.get_out_size(img_size, patch_size)
        self.out_size1 = self.get_out_size(img_size, patch_size*2)

        # block
        self.emb = PatchEmbedLayer(in_channels, emb_dims[0], patch_size, img_size, has_token)

        self.before_block0 = nn.Sequential()
        for _ in range(num_blocks[0]):
            self.before_block0.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.down_sample0 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(emb_dims[0], emb_dims[1], kernel_size=1),
            nn.LayerNorm([emb_dims[1], self.out_size1[0], self.out_size1[1]]),
            nn.GELU()
        )

        self.before_block1 = nn.Sequential()
        for _ in range(num_blocks[1]):
            self.before_block1.append(NormalEncoderBlock(emb_dims[1], heads[1], hidden_dims[1], drop_rate))

        self.down_sample1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(emb_dims[1], emb_dims[2], kernel_size=1),
            nn.LayerNorm([emb_dims[2], self.out_size1[0]//2, self.out_size1[1]//2]),
            nn.GELU()
        )

        self.bottom_block = nn.Sequential()
        for _ in range(num_blocks[2]):
            self.bottom_block.append(NormalEncoderBlock(emb_dims[2], heads[2], hidden_dims[2], drop_rate))

        self.up_sample1 = UpLayer(emb_dims[2], emb_dims[1], self.out_size1)

        self.after_block1 = nn.Sequential()
        for _ in range(num_blocks[3]):
            self.after_block1.append(NormalEncoderBlock(emb_dims[1], heads[1], hidden_dims[1], drop_rate))

        self.up_sample0 = UpLayer(emb_dims[1], emb_dims[0], self.out_size0)

        self.after_block0 = nn.Sequential()
        for _ in range(num_blocks[4]):
            self.after_block0.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.final = nn.Sequential(
            nn.Conv2d(emb_dims[0], emb_dims[0], kernel_size=1),
            nn.LayerNorm([emb_dims[0], self.out_size0[0], self.out_size0[1]]),
            nn.GELU()
        )

    def get_out_size(self, img_size, patch_size):
        return (img_size[0]//patch_size, img_size[1]//patch_size)

    def forward(self, x:torch.Tensor):
        B = x.shape[0]

        # patch embedding
        x = self.emb(x)
        # transformer before downsample
        x = self.before_block0(x)
        x = self.reshape_feat(x, B)
        # downsample
        x_m = self.down_sample0(x)
        x_m = self.flat_feat(x_m)
        # add token
        x_m = self.before_block1(x_m)
        x_m = self.reshape_feat(x_m, B, 2, 1)

        x_s = self.down_sample1(x_m)
        x_s = self.flat_feat(x_s)
        # transformer
        x_s = self.bottom_block(x_s)
        # reshape
        x_s = self.reshape_feat(x_s, B, 4, 2)
        x_s = self.up_sample1(x_s)

        x_m = x_m + x_s
        x_m = self.flat_feat(x_m)
        x_m = self.after_block1(x_m)
        x_m = self.reshape_feat(x_m, B, 2, 1)
        x_m = self.up_sample0(x_m)
        
        x = x + x_m
        x = self.flat_feat(x)
        x = self.after_block0(x)
        x = self.reshape_feat(x, B)
        x = self.final(x)

        return tuple([x])

    def reshape_feat(self, feat:torch.Tensor, b, factor=1, id=0):
        return feat.reshape((b, self.out_size0[0]//factor, self.out_size0[1]//factor, self.emb_dims[id])).permute(0, 3, 1, 2)
    
    def flat_feat(self, feat:torch.Tensor):
        feat = feat.flatten(2)
        feat = feat.transpose(1, 2)
        return feat