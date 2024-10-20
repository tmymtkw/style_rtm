import torch
import torch.nn as nn

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from ..utils import PointwiseConvLayer, DWConvBlock, UpsampleLayer, PatchEmbedLayer, NormalEncoderBlock
from .uvit import UpLayer

class PatchDownSample(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 feat_size,
                 scale = 2,
                 has_token = True):
        super().__init__()
        # var
        self.in_dim = in_dim
        self.has_token = has_token
        self.feat_size = feat_size
        # module
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=scale, stride=scale, padding=0)
        self.act = nn.GELU()
        self.token_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x:torch.Tensor):
        token = x[:, :self.has_token]
        x = x[:, self.has_token:]
        x = x.reshape((x.shape[0], *self.feat_size, self.in_dim)).permute(0, 3, 1, 2)

        token = self.token_proj(token)
        x = self.conv(x)        
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([token, x], dim=1)
        
        return x
    
class PatchUpSample(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 feat_size,
                 has_token = True,
                 with_token = True):
        super().__init__()
        # var
        self.in_dim = in_dim
        self.has_token = has_token
        self.feat_size = feat_size
        self.with_token = with_token
        # module
        self.upsample = UpLayer(in_dim, out_dim, feat_size)
        self.token_proj = nn.Linear(in_dim, out_dim, False)

    def forward(self, x:torch.Tensor):
        token = x[:, :self.has_token]
        x = x[:, self.has_token:]
        x = x.reshape((x.shape[0], self.feat_size[0]//2, self.feat_size[1]//2, self.in_dim)).permute(0, 3, 1, 2)

        x = self.upsample(x)
        if not self.with_token:
            return x
        token = self.token_proj(token)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([token, x], dim=1)

        return x
    
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 drop_rate):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.final = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.final(x)
        return x
    
class StyleEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 drop_rate,
                 ):
        super().__init__()
        self.conv0 = ConvBlock(in_channels, 8, 5, 2, 2, drop_rate)

        self.pool = nn.AvgPool2d(2)

        self.conv1 = nn.Sequential(
            ConvBlock(8, 16, 5, 2, 2, drop_rate),
            DWConvBlock(16, 32)
        )

        self.conv2 = nn.Sequential(
            ConvBlock(16, 32, 5, 2, 2, drop_rate),
            DWConvBlock(32, 64)
        )


    def forward(self, x:torch.Tensor):
        x = self.conv0(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn = self.linear(self.avg_pool(x)) + self.linear(self.max_pool(x))
        attn = self.act(attn)
        return attn

@MODELS.register_module()
class StyleNet2(BaseBackbone):
    def __init__(self,
                 in_channels,
                 emb_dims = (96, 384),
                 patch_size = 8,
                 img_size = (256, 192),
                 out_size = (64, 48),
                 hidden_dims = (96*2, 384*4),
                 num_blocks = (3, 6, 3),
                 heads = (6, 12),
                 drop_rate = 0.1,
                 has_token = True
                 ):
        super().__init__()
        # var
        self.emb_dims = emb_dims
        self.out_dim = emb_dims[0] // 2
        self.img_size = img_size
        self.out_size = out_size
        # module
        self.emb = PatchEmbedLayer(in_channels, emb_dims[0], patch_size, img_size, has_token)

        self.m_front = nn.Sequential()
        for _ in range(num_blocks[0]):
            self.m_front.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        self.down = PatchDownSample(emb_dims[0], emb_dims[1], (out_size[0]//2, out_size[1]//2))

        self.bot = nn.Sequential()
        for _ in range(num_blocks[1]):
            self.bot.append(NormalEncoderBlock(emb_dims[1], heads[1], hidden_dims[1], drop_rate))

        self.up0 = PatchUpSample(emb_dims[1], emb_dims[0], (out_size[0]//2, out_size[1]//2))

        self.m_behind = nn.Sequential()
        for _ in range(num_blocks[2]):
            self.m_behind.append(NormalEncoderBlock(emb_dims[0], heads[0], hidden_dims[0], drop_rate))

        # self.up1 = PatchUpSample(emb_dims[0], self.out_dim, out_size, with_token=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(emb_dims[0], self.out_dim, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([self.out_dim, out_size[0], out_size[1]]),
            nn.Conv2d(self.out_dim, self.out_dim*2, 1),
            nn.GELU(),
            nn.Conv2d(self.out_dim*2, self.out_dim, 1)
        )

        self.style_encoder = StyleEncoder(in_channels, drop_rate)

        self.attn = ChannelAttention(32, emb_dims[0]*2, emb_dims[0])

    def get_size(self, ratio:int):
        return (self.img_size[0]//ratio, self.img_size[1]//ratio)

    def forward(self, x:torch.Tensor):
        feat_style = self.style_encoder(x)
        attn = self.attn(feat_style).unsqueeze(1)

        feat = self.emb(x)
        feat = feat * attn
        
        feat = self.m_front(feat)

        feat_s = self.down(feat)
        feat_s = self.bot(feat_s)

        feat = feat + self.up0(feat_s)
        feat = self.m_behind(feat)

        feat = feat[:, 1:]
        feat = feat.reshape((x.shape[0], self.out_size[0]//2, self.out_size[1]//2, self.emb_dims[0])).permute(0, 3, 1, 2)
        feat = self.final(feat)

        return tuple([feat_style, feat])