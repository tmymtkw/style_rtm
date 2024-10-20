import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.utils.rtmcc_block import RTMCCBlock
from mmpose.registry import  MODELS
from ..base_head import BaseHead

@MODELS.register_module()
class MixHead(BaseHead):
    def __init__(self):
        super(MixHead, self).__init__()
        # simcc
        self.gau = RTMCCBlock()
        self.mlp_x = nn.Linear()
        self.mlp_y = nn.Linear()

        # heatmap
        self.deconv_layer = nn.ConvTranspose2d()

    def forward(self, x:tuple[torch.Tensor, torch.Tensor]):
        feat_kpt, feat_img = x

        feat_kpt = self.gau(feat_kpt) + feat_kpt
        pred_x = self.mlp_x(feat_kpt)
        pred_y = self.mlp_y(feat_kpt)

        feat_img = self.deconv_layer(feat_img)

        
