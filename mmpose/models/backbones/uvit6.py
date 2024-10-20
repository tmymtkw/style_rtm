import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .myvit import PatchEmbedLayer, NormalEncoderBlock
from .uvit import UpLayer
from ..utils import RTMCCBlock

class DownSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 normalized_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 2, 2)
        self.norm = nn.LayerNorm(normalized_shape)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

@MODELS.register_module()
class UViT6(BaseBackbone):
    def __init__(self,
                 in_channels = 3,
                 emb_dims = (24, 96, 384),
                 hidden_dims = (24*4, 96*4, 384*4),
                 patch_size = 4,
                 img_size = (256, 192),
                 num_blocks = (1, 1, 1, 1, 1),
                 drop_rate = 0.1,
                 has_token = True):
        super().__init__()
        # var
        self.emb_dims = emb_dims
        self.out_size = self.get_out_size(img_size, patch_size)
        self.num_token = self.out_size[0]*self.out_size[1]
        self.has_token = has_token

        # block
        self.emb = PatchEmbedLayer(in_channels, emb_dims[0], patch_size, img_size, has_token)


        self.before_block0 = nn.Sequential()
        for _ in range(num_blocks[0]):
            self.before_block0.append(RTMCCBlock(num_token=self.num_token+1, 
                                                 in_token_dims=emb_dims[0], 
                                                 out_token_dims=emb_dims[0],
                                                 s=hidden_dims[0],
                                                 dropout_rate=drop_rate))

        self.down_sample0 = DownSampleBlock(emb_dims[0], emb_dims[1], [emb_dims[1], self.out_size[0]//2, self.out_size[1]//2])

        self.down_linear0 = nn.Linear(emb_dims[0], emb_dims[1], bias=False)


        self.before_block1 = nn.Sequential()
        for _ in range(num_blocks[1]):
            self.before_block1.append(RTMCCBlock(num_token=self.num_token//4+1,
                                                 in_token_dims=emb_dims[1],
                                                 out_token_dims=emb_dims[1],
                                                 s=hidden_dims[1],
                                                 dropout_rate=drop_rate))

        self.down_sample1 = DownSampleBlock(emb_dims[1], emb_dims[2], [emb_dims[2], self.out_size[0]//4, self.out_size[1]//4])

        self.down_linear1 = nn.Linear(emb_dims[1], emb_dims[2], bias=False)


        self.bottom_block = nn.Sequential()
        for _ in range(num_blocks[2]):
            self.bottom_block.append(RTMCCBlock(num_token=self.num_token//16+1,
                                                in_token_dims=emb_dims[2],
                                                out_token_dims=emb_dims[2],
                                                s=hidden_dims[2],
                                                dropout_rate=drop_rate))


        self.up_sample1 = UpLayer(emb_dims[2], emb_dims[1], (self.out_size[0]//2, self.out_size[1]//2))

        self.up_linear1 = nn.Linear(emb_dims[2], emb_dims[1], bias=False)

        self.after_block1 = nn.Sequential()
        for _ in range(num_blocks[3]):
            self.after_block1.append(RTMCCBlock(num_token=self.num_token//4+1,
                                                in_token_dims=emb_dims[1],
                                                out_token_dims=emb_dims[1],
                                                s=hidden_dims[1],
                                                dropout_rate=drop_rate))


        self.up_sample0 = UpLayer(emb_dims[1], emb_dims[0], self.out_size)

        self.up_linear0 = nn.Linear(emb_dims[1], emb_dims[0], bias=False)

        self.after_block0 = nn.Sequential()
        for _ in range(num_blocks[4]):
            self.after_block0.append(RTMCCBlock(num_token=self.out_size[0]*self.out_size[1]+1,
                                                in_token_dims=emb_dims[0],
                                                out_token_dims=emb_dims[0],
                                                s=hidden_dims[0],
                                                dropout_rate=drop_rate))


        self.final = nn.GELU()

    def get_out_size(self, img_size, patch_size):
        return (img_size[0]//patch_size, img_size[1]//patch_size)

    def forward(self, x:torch.Tensor):
        B = x.shape[0]

        # patch embedding
        x = self.emb(x)
        # before 0
        x = self.before_block0(x)
        token = x[:, :self.has_token]
        x = x[:, self.has_token:]
        x = self.reshape_feat(x, B)
        token_m = self.down_linear0(token)
        x_m = self.down_sample0(x)
        x_m = self.flat_feat(x_m)
        x_m = torch.cat([token_m, x_m], dim=1)

        # before 1
        x_m = self.before_block1(x_m)
        token_m = x_m[:, :self.has_token]
        x_m = x_m[:, self.has_token:]
        x_m = self.reshape_feat(x_m, B, 2, 1)
        token_s = self.down_linear1(token_m)
        x_s = self.down_sample1(x_m)
        x_s = self.flat_feat(x_s)
        x_s = torch.cat([token_s, x_s], dim=1)

        # bottom
        x_s = self.bottom_block(x_s)
        token_s = x_s[:, :self.has_token]
        x_s = x_s[:, self.has_token:]
        x_s = self.reshape_feat(x_s, B, 4, 2)
        x_s = self.up_sample1(x_s)
        token_m = token_m + self.up_linear1(token_s)
        x_m = x_m + x_s
        x_m = self.flat_feat(x_m)
        x_m = torch.cat([token_m, x_m], dim=1)

        x_m = self.after_block1(x_m)
        token_m = x_m[:, :self.has_token]
        x_m = x_m[:, self.has_token:]
        x_m = self.reshape_feat(x_m, B, 2, 1)
        x_m = self.up_sample0(x_m)
        token = token + self.up_linear0(token_m)
        x = x + x_m
        x = self.flat_feat(x)
        x = torch.cat([token, x], dim=1)

        x = self.after_block0(x)
        x = x[:, self.has_token:]
        x = self.reshape_feat(x, B)
        x = self.final(x)

        return tuple([x])

    def reshape_feat(self, feat:torch.Tensor, b, factor=1, id=0):
        return feat.reshape((b, self.out_size[0]//factor, self.out_size[1]//factor, self.emb_dims[id])).permute(0, 3, 1, 2)
    
    def flat_feat(self, feat:torch.Tensor):
        feat = feat.flatten(2)
        feat = feat.transpose(1, 2)
        return feat