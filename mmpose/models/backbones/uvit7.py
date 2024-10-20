import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .myvit import PatchEmbedLayer, NormalEncoderBlock
from .uvit import UpLayer

@MODELS.register_module()
class UViT7(BaseBackbone):
    def __init__(self,
                 in_channels = 3,
                 emb_dims = (96, 384, 24),
                 hidden_dims = (96*2, 384*2),
                 patch_size = 8,
                 img_size = (256, 192),
                 heads = (6, 12, 3),
                 num_blocks = (3, 6, 3, 2),
                 drop_rate = 0.1,
                 has_token = True):
        super().__init__()
        # var
        self.emb_dims = emb_dims
        self.out_size = self.get_out_size(img_size, patch_size)
        self.has_token = has_token

        # block
        self.emb = PatchEmbedLayer(in_channels, emb_dims[0], patch_size, img_size, has_token)

        self.before_block = nn.Sequential()
        for _ in range(num_blocks[0]):
            self.before_block.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.linear0 = nn.Linear(emb_dims[0], emb_dims[1], bias=False)

        self.down_sample = nn.Sequential(
            nn.Conv2d(emb_dims[0], emb_dims[1], kernel_size=2, stride=2),
            nn.LayerNorm([emb_dims[1], self.out_size[0]//2, self.out_size[1]//2]),
            nn.GELU()
        )

        self.bottom_block = nn.Sequential()
        for _ in range(num_blocks[1]):
            self.bottom_block.append(NormalEncoderBlock(emb_dims[1], heads[2], hidden_dims[1], drop_rate))

        self.up_sample0 = UpLayer(emb_dims[1], emb_dims[0], self.out_size)

        self.linear1 = nn.Linear(emb_dims[1], emb_dims[0], bias=False)

        self.after_block = nn.Sequential()
        for _ in range(num_blocks[2]):
            self.after_block.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.up_sample1 = UpLayer(emb_dims[0], emb_dims[2], self.get_out_size(img_size, patch_size//2))

        self.final = nn.Sequential()
        for _ in range(num_blocks[3]):
            self.final.append(NormalEncoderBlock(emb_dims[2], heads[2], emb_dims[2]*2, drop_rate))

        self.act = nn.GELU()

    def get_out_size(self, img_size, patch_size):
        return (img_size[0]//patch_size, img_size[1]//patch_size)

    def forward(self, x:torch.Tensor):
        B = x.shape[0]

        # patch embedding
        x = self.emb(x)
        # transformer before downsample
        x = self.before_block(x)
        token = x[:, :self.has_token]
        x = x[:, self.has_token:]
        x = self.reshape_feat(x, B)
        token = self.linear0(token)
        # downsample
        x_s = self.down_sample(x)
        x_s = self.flat_feat(x_s)
        # add token
        x_s = torch.cat([token, x_s], dim=1)
        # transformer
        x_s = self.bottom_block(x_s)
        # reshape
        token = x_s[:, :self.has_token]
        token = self.linear1(token)
        x_s = x_s[:, self.has_token:]
        x_s = self.reshape_feat(x_s, B, 2, 1)
        x_s = self.up_sample0(x_s)

        x = x + x_s
        x = self.act(x)
        x = self.flat_feat(x)
        x = torch.cat([token, x], dim=1)
        x = self.after_block(x)

        x = x[:, self.has_token:]
        x = self.reshape_feat(x, B)
        x = self.up_sample1(x)
        x = self.flat_feat(x)
        x = self.final(x)
        x = self.reshape_feat(x, B, 0.5, 2)

        return tuple([x])

    def reshape_feat(self, feat:torch.Tensor, b, factor=1, id=0):
        return feat.reshape((b, int(self.out_size[0]//factor), int(self.out_size[1]//factor), self.emb_dims[id])).permute(0, 3, 1, 2)
    
    def flat_feat(self, feat:torch.Tensor):
        feat = feat.flatten(2)
        feat = feat.transpose(1, 2)
        return feat