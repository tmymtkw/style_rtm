# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .cpm import CPM
from .hourglass import HourglassNet
from .hourglass_ae import HourglassAENet
from .hrformer import HRFormer
from .hrnet import HRNet
from .litehrnet import LiteHRNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mspn import MSPN
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .rsn import RSN
from .scnet import SCNet
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin import SwinTransformer
from .tcn import TCN
from .v2v_net import V2VNet
from .vgg import VGG
from .vipnas_mbv3 import ViPNAS_MobileNetV3
from .vipnas_resnet import ViPNAS_ResNet
from .vision_transformer_custom import VisionTransformerU2
from .halfvit import HalfVit, HalfCatVit, NormalVit
from .myvit import MyVit
from .myvit2 import MyVit2
from .myvit3 import MyVit3
from .uvit import UViT
from .uvit2 import UViT2
from .uvit3 import UViT3
from .myvit4 import MyViT4
from .uvit4 import UViT4
from .uvit5 import UViT5
from .uvit6 import UViT6
from .uvit7 import UViT7
from .uvit8 import UViT8
from .uvit9 import UViT9
from .uvit10 import UViT10
from .stylenet import StyleNet
from .stylenet2 import StyleNet2
from .stylenet3 import StyleNet3
from .stylenet4 import StyleNet4
from .cspnext2 import CSPNeXt2
from .cspnext3 import CSPNeXt3
from .cspnext4 import CSPNeXt4
from .cspnextiir import CSPNeXtIIR
from .cspnextyle import CSPNeXtyle
from .cspnextyle2 import CSPNeXtyle2
from .cspnextyle3 import CSPNeXtyle3
from .cspnextyle4 import CSPNeXtyle4
from .cspnextyle5 import CSPNeXtyle5
from .cspnextyle5abl import CSPNeXtyle5Abl
from .cspnextyle6 import CSPNeXtyle6
from .cspara import CSPaRallel
from .cspara2 import CSPaRallel2
from .dualcspnext import DualCSPNeXt

__all__ = [
    'AlexNet', 'HourglassNet', 'HourglassAENet', 'HRNet', 'MobileNetV2',
    'MobileNetV3', 'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SCNet',
    'SEResNet', 'SEResNeXt', 'ShuffleNetV1', 'ShuffleNetV2', 'CPM', 'RSN',
    'MSPN', 'ResNeSt', 'VGG', 'TCN', 'ViPNAS_ResNet', 'ViPNAS_MobileNetV3',
    'LiteHRNet', 'V2VNet', 'HRFormer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'SwinTransformer', 'VisionTransformerU2',
    'HalfVit', 'HalfCatVit', 'MyVit', 'NormalVit', 'MyVit2', 'MyVit3', 'UViT', 
    'UViT2', 'UViT3', 'MyViT4', 'UViT4', 'UViT5', 'StyleNet', 'UViT6', 'UViT7',
    'StyleNet2', 'UViT8', 'UViT9', 'CSPNeXt2', 'CSPNeXt3', 'CSPNeXt4', 'UViT10',
    'CSPNeXtIIR', 'StyleNet3', 'CSPNeXtyle', 'CSPaRallel', 'CSPaRallel2', 'StyleNet4',
    'CSPNeXtyle2', 'DualCSPNeXt', 'CSPNeXtyle3', 'CSPNeXtyle4', 'CSPNeXtyle5',
    'CSPNeXtyle6', 'CSPNeXtyle5Abl'
]
