import math
from typing import Sequence, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers import CSPLayer
from mmdet.models.backbones.csp_darknet import SPPBottleneck

from ..utils import ReConvNeXt2

@MODELS.register_module()
class CSPaRallel2(BaseModule):
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch: str = 'P5',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 use_depthwise: bool = False,
                 expand_ratio: float = 0.5,
                 arch_ovewrite: dict = None,
                 spp_kernel_sizes: Sequence[int] = (5, 9, 13),
                 channel_attention: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = dict(type='Kaiming',
                                                 layer='Conv2d',
                                                 a=math.sqrt(5),
                                                 distribution='uniform',
                                                 mode='fan_in',
                                                 nonlinearity='leaky_relu')):
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        begin_channels = int(arch_setting[0][0] * widen_factor // 2)

        self.stem = nn.Sequential(
            ReConvNeXt2(
                3,
                begin_channels//2,
                begin_channels//2,
                kernel_size=3,
                stride=2),

            ReConvNeXt2(
                in_channels=begin_channels//2,
                hidden_channels=begin_channels,
                out_channels=begin_channels,
                kernel_size=3,
                stride=1,
                shortcut=False),
                
            ReConvNeXt2(in_channels=begin_channels,
                       hidden_channels=begin_channels*2,
                       out_channels=begin_channels*2,
                       kernel_size=3,
                       stride=1,
                       shortcut=False),


        )
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = ReConvNeXt2(
                in_channels=in_channels,
                hidden_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2
            )
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

        pass

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            #print(layer_name)
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)