import torch
import torch.nn as nn

from mmpretrain.models.backbones import VisionTransformer
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

class EdgeConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_horizon = torch.tensor([[-1, 0, 1], 
                                            [-1, 0, 1], 
                                            [-1, 0, 1]], 
                                            device=torch.device("cuda"),
                                            dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_horizon.requires_grad = False
        self.kernel_vertical = torch.tensor([[1, 1, 1], 
                                             [0, 0, 0], 
                                             [-1, -1, -1]], 
                                             device=torch.device("cuda"),
                                             dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_vertical.requires_grad = False
        self.gray_kernel = torch.tensor([0.299, 0.587, 0.114], 
                                        device=torch.device("cuda"),
                                        dtype=torch.float32).reshape(3, 1, 1)  # color -> gray kernel
        self.gray_kernel.requires_grad = False

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=4, padding=2, bias=True),
                                    nn.BatchNorm2d(num_features=1),
                                    nn.ReLU())
        # self.layer2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2, bias=True),
        #                             nn.BatchNorm2d(num_features=1),
        #                             nn.ReLU())

    def getedge(self, color):
        gray = torch.sum(color * self.gray_kernel, dim=1, keepdim=True)  # grayscale image [B, 1, H, W]
        # print(gray.shape)
        # エッジ検出
        edge = nn.functional.conv2d(gray, self.kernel_horizon, padding=1) + nn.functional.conv2d(gray, self.kernel_vertical, padding=1)
        # print(edge.shape)
        return edge
    
    def forward(self, x):
        x_edge = self.getedge(x)
        edge1 = self.layer1(x_edge)
        # edge2 = self.layer2(edge1)
        return tuple([edge1])

@MODELS.register_module()
class EdgeViT(BaseBackbone):
    def __init__(self, 
                arch={
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 384 * 4
                },
                img_size=(256, 192),
                patch_size=16,
                qkv_bias=True,
                drop_path_rate=0.1,
                with_cls_token=True,
                out_type='featmap',
                patch_cfg=dict(padding=2),
                init_cfg=dict(
                type='Pretrained',
                checkpoint='checkpoints/mae_pretrain_vit_small.pth')
                ):
        super().__init__()
        self.vit = VisionTransformer(arch=arch, 
                                     img_size=img_size, 
                                     patch_size=patch_size,
                                     qkv_bias=qkv_bias,
                                     drop_path_rate=drop_path_rate,
                                     with_cls_token=with_cls_token,
                                     out_type=out_type,
                                     patch_cfg=patch_cfg,
                                     init_cfg=init_cfg)
        self.edgeconv = EdgeConv()
        
    def forward(self, x):
        out = self.vit(x)
        edge = self.edgeconv(x)
        return tuple([out, edge])