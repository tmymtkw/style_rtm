# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .simcc_head import SimCCHead
from .lightrtmcc import LightRTMCCHead
from .rtmcc_head2 import RTMCCHead2
from .rtmcc_head3 import RTMCCHead3
from .rtmcc_head4 import RTMCCHead4

__all__ = ['SimCCHead', 'RTMCCHead', 'LightRTMCCHead', 'RTMCCHead2', 'RTMCCHead3',
           'RTMCCHead4']
