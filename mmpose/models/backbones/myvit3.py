import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

from .myvit import PatchEmbedLayer, NormalEncoderBlock
from .myvit2 import DecoderBlock

@MODELS.register_module()
class MyVit3(BaseBackbone):
    def __init__(self,
                 in_channels:int,
                 emb_dim:int,
                 patch_size:int,
                 img_size = (192, 256),
                 head:int = 12,
                 hidden_dim:int = 384,
                 drop_rate:float = 0.,
                 has_token:bool = True,
                 num_kpt:int = 17):
        super().__init__()
        self.kpt_token = torch.Tensor([i for i in range(num_kpt)]).to(device="cuda", dtype=torch.int32)

        self.emb_img = PatchEmbedLayer(in_channels,
                                        emb_dim,
                                        patch_size,
                                        img_size,
                                        has_token)
        
        self.emb_kpt = nn.Embedding(num_kpt, emb_dim)
        
        self.enc0 = NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.enc1 = NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.enc2 = NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.enc3 = NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.enc4 = NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.enc5 = NormalEncoderBlock(emb_dim, head, hidden_dim, drop_rate)

        self.dec0 = DecoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.dec1 = DecoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.dec2 = DecoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.dec3 = DecoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.dec4 = DecoderBlock(emb_dim, head, hidden_dim, drop_rate)
        self.dec5 = DecoderBlock(emb_dim, head, hidden_dim, 0.0)

    def forward(self, x:torch.Tensor):
        batch_size = x.shape[0]
        feat_img = self.emb_img(x)
        feat_kpt = self.emb_kpt(self.kpt_token)
        feat_kpt = feat_kpt.repeat(repeats=(batch_size, 1, 1))

        feat_img = self.enc0(feat_img)
        feat_kpt = self.dec0(feat_kpt, feat_img)

        feat_img = self.enc1(feat_img)
        feat_kpt = self.dec1(feat_kpt, feat_img)

        feat_img = self.enc2(feat_img)
        feat_kpt = self.dec2(feat_kpt, feat_img)

        feat_img = self.enc3(feat_img)
        feat_kpt = self.dec3(feat_kpt, feat_img)

        feat_img = self.enc4(feat_img)
        feat_kpt = self.dec4(feat_kpt, feat_img)

        feat_img = self.enc5(feat_img)
        feat_kpt = self.dec5(feat_kpt, feat_img)


        return tuple([feat_kpt])