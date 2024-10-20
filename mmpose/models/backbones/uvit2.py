import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .myvit import PatchEmbedLayer, NormalEncoderBlock
from .uvit import UpLayer

@MODELS.register_module()
class UViT2(BaseBackbone):
    def __init__(self,
                 in_channels:int,
                 emb_dims:tuple[int, int, int],
                 hidden_dims:tuple[int, int, int],
                 patch_sizes:tuple[int, int, int] = (16, 8),
                 img_size = (256, 192),
                 head:int = 12,
                 num_blocks:tuple[int, int, int] = (3, 2),
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

        self.up0 = UpLayer(emb_dims[0], emb_dims[1], self.out_size0)

        self.up1 = UpLayer(emb_dims[1]*2, emb_dims[1]*2, self.out_size1)

        self.final = nn.Conv2d(emb_dims[1]*2, emb_dims[1], kernel_size=1)

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
        feat1 = torch.cat((feat0, feat1), dim=1)
        feat1 = self.up1(feat1)

        feat1 = self.final(feat1)

        # debug
        # for o in out:
        #     print(o.shape)

        return tuple([feat1])
    
    def reshape_feat(self, feat:torch.Tensor, batch_size:int, out_size:tuple, id:int):
        img_feat = feat[:, self.has_token:]
        return img_feat.reshape((batch_size, *out_size, self.emb_dims[id])).permute(0, 3, 1, 2)


