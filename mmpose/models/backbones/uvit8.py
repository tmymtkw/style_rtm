import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .uvit import UpLayer
from ..utils import DecoderBlock, PatchEmbedLayer, NormalEncoderBlock

@MODELS.register_module()
class UViT8(BaseBackbone):
    def __init__(self,
                 in_channels = 3,
                 emb_dims = (96, 384),
                 hidden_dims = (96*2, 384*4),
                 patch_size = 8,
                 img_size = (256, 192),
                 heads = (6, 12),
                 num_blocks = (3, 6, 3),
                 bot_block = 6,
                 num_attn = 3,
                 drop_rate = 0.1,
                 has_token = True):
        super().__init__()
        # var
        self.emb_dims = emb_dims
        self.out_size = self.get_out_size(img_size, patch_size)
        self.has_token = has_token
        self.num_attn = num_attn

        # block
        self.emb = PatchEmbedLayer(in_channels, emb_dims[0], patch_size, img_size, has_token)

        self.before_block = nn.Sequential()
        for _ in range(num_blocks[0]):
            self.before_block.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.linear = nn.Linear(emb_dims[0], emb_dims[1], bias=False)

        self.down_sample = nn.Sequential(
            nn.Conv2d(emb_dims[0], emb_dims[1], kernel_size=2, stride=2),
            nn.LayerNorm([emb_dims[1], self.out_size[0]//2, self.out_size[1]//2]),
            nn.GELU()
        )

        # self.mid_block = nn.Sequential()
        # for _ in range(num_blocks[1]):
        #     self.mid_block.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.bottom_block = nn.Sequential()
        for _ in range(bot_block):
            self.bottom_block.append(NormalEncoderBlock(emb_dims[1], heads[1], hidden_dims[1], drop_rate))

        self.cat_attn = nn.ModuleList()
        for _ in range(num_attn):
            self.cat_attn.append(DecoderBlock(emb_dims[0], heads[0], hidden_dims[1], drop_rate))

        self.up_sample = UpLayer(emb_dims[1], emb_dims[0], self.out_size)

        self.after_block = nn.Sequential()
        for _ in range(num_blocks[2]):
            self.after_block.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.final = nn.Sequential(
            nn.ConvTranspose2d(emb_dims[0], emb_dims[0], kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([emb_dims[0], self.out_size[0]*2, self.out_size[1]*2]),
            nn.GELU()
        )

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
        token_s = self.linear(token)
        # downsample
        x_s = self.down_sample(x)
        x_s = self.flat_feat(x_s)
        # add token
        x_s = torch.cat([token_s, x_s], dim=1)
        # transformer
        x_s = self.bottom_block(x_s)
        # reshape
        x_s = x_s[:, self.has_token:]
        x_s = self.reshape_feat(x_s, B, 2, 1)
        x_s = self.up_sample(x_s)
        x_s = self.flat_feat(x_s)

        x = self.flat_feat(x)
        # x = self.mid_block(x)

        for i in range(self.num_attn):
            x = self.cat_attn[i](x, x_s)

        x = self.after_block(x)
        x = self.reshape_feat(x, B)
        x = self.final(x)

        return tuple([x])

    def reshape_feat(self, feat:torch.Tensor, b, factor=1, id=0):
        return feat.reshape((b, self.out_size[0]//factor, self.out_size[1]//factor, self.emb_dims[id])).permute(0, 3, 1, 2)
    
    def flat_feat(self, feat:torch.Tensor):
        feat = feat.flatten(2)
        feat = feat.transpose(1, 2)
        return feat