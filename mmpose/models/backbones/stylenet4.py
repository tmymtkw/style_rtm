import torch
import torch.nn as nn

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from ..utils import PointwiseConvLayer, DWConvBlock, UpsampleLayer, PatchEmbedLayer, NormalEncoderBlock, DecoderBlock
from .uvit import UpLayer
from .stylenet3 import StyleEncoder

@MODELS.register_module()
class StyleNet4(BaseBackbone):
    def __init__(self,
                 in_channels,
                 emb_dims = 96,
                 patch_size = 8,
                 img_size = (256, 192),
                 out_size = (64, 48),
                 hidden_dims = 96*4,
                 num_blocks = (3, 6, 3),
                 heads = (6, 12),
                 drop_rate = 0.1,
                 reduction=16,
                 cls_num=20,
                 cls_num_extra=20,
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

        self.style_encoder = StyleEncoder(in_channels, 
                                          hidden_dims=(8, 16, 32, 64), 
                                          cls_num=cls_num+cls_num_extra, reduction=1024,  drop_rate=drop_rate, inplace=False)

        self.style_emb_channel = nn.Sequential(
            nn.Linear(cls_num+cls_num_extra, emb_dims//reduction),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(emb_dims//reduction, emb_dims),
            nn.Sigmoid())
        
        self.space_dim = self.get_space_dim(self.get_size(ratio=8))
        self.style_emb_space = nn.Sequential(
            nn.Linear(cls_num+cls_num_extra, self.space_dim//reduction),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.space_dim//reduction, self.space_dim+has_token),
            nn.Sigmoid())
        
    def get_space_dim(self, size: tuple):
        return size[0] * size[1]

    def get_size(self, ratio:int):
        return (self.img_size[0]//ratio, self.img_size[1]//ratio)

    def forward(self, x:torch.Tensor):
        B = x.shape[0]

        feat_style = self.style_encoder(x)

        feat = self.emb(x)
        feat_ = feat

        # channel attention
        feat_cls_channel = self.style_emb_channel(feat_style)
        feat_cls_channel = feat_cls_channel.unsqueeze(1)
        feat = feat * feat_cls_channel
        # spatial attention
        feat_cls_space = self.style_emb_space(feat_style)
        feat_cls_space = feat_cls_space.unsqueeze(2)
        feat = feat * feat_cls_space

        feat = feat + feat_
        feat = self.m_front(feat)
        token = feat[:, :1]
        feat = feat[:, 1:]

        feat_s = self.reshape_feat(feat, B, 1)
        feat_s = self.down(feat_s)
        feat_s = self.flat_feat(feat_s)
        feat_s = torch.cat([token, feat_s], dim=1)
        feat_s = self.bot(feat_s)
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