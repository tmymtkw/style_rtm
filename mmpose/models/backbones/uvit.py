import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .myvit import PatchEmbedLayer, NormalEncoderBlock

class UpLayer(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 feat_size:tuple[int, int]):
        super().__init__()
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm([out_channels, *feat_size])
        self.relu = nn.GELU()

    def forward(self, x:torch.Tensor):
        feat = self.pw_conv(x)
        feat = F.interpolate(feat, scale_factor=2., mode="bilinear")
        feat = self.norm(feat)
        feat = self.relu(feat)
        return feat

@MODELS.register_module()
class UViT(BaseBackbone):
    def __init__(self,
                 in_channels:int,
                 emb_dims:tuple[int, int],
                 hidden_dims:tuple[int, int],
                 patch_sizes:tuple[int, int] = (16, 8),
                 img_size = (256, 192),
                 head:int = 12,
                 num_blocks:tuple[int, int] = (3, 2),
                 drop_rate:float = 0.1,
                 has_token:bool = True
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dims = emb_dims
        self.hidden_dims = hidden_dims
        self.patch_sizes = patch_sizes
        self.img_size = img_size
        self.head = head
        self.num_blocks = num_blocks
        self.drop_rate = drop_rate
        self.has_token = has_token
        
        # stage 0
        self.out_size0 = self.get_out_size(patch_sizes[0])
        self.stage0 = self.get_layer(0)
        
        # stage 1
        self.out_size1 = self.get_out_size(patch_sizes[1])
        self.stage1 = self.get_layer(1)

        self.up0 = UpLayer(emb_dims[0], emb_dims[1], self.out_size1)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(emb_dims[1], emb_dims[1], kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([emb_dims[1], self.out_size1[0]*2, self.out_size1[1]*2]),
            nn.GELU()
        )

    def get_emb_layer(self, id:int):
        return PatchEmbedLayer(self.in_channels, self.emb_dims[id], self.patch_sizes[id], self.img_size, self.has_token)
    
    def get_out_size(self, patch_size:int):
        return (self.img_size[0]//patch_size, self.img_size[1]//patch_size)
    
    def get_layer(self, stage_id:int):
        layer = nn.Sequential()

        layer.append(self.get_emb_layer(stage_id))
        for _ in range(self.num_blocks[stage_id]):
            layer.append(NormalEncoderBlock(self.emb_dims[stage_id], self.head, self.hidden_dims[stage_id], self.drop_rate))
        
        return layer
    
    def forward(self, x:torch.Tensor):
        B = x.shape[0]

        feat0 = self.stage0(x)
        feat0 = self.reshape_feat(feat0, B, self.out_size0, 0)
        feat0 = self.up0(feat0)

        feat1 = self.stage1(x)
        feat1 = self.reshape_feat(feat1, B, self.out_size1, 1)
        out = feat0 + feat1

        out = self.final(out)

        # debug
        # for o in out:
        #     print(o.shape)

        return tuple([out])
    
    def reshape_feat(self, feat:torch.Tensor, batch_size:int, out_size:tuple, id:int):
        img_feat = feat[:, self.has_token:]
        return img_feat.reshape((batch_size, *out_size, self.emb_dims[id])).permute(0, 3, 1, 2)


