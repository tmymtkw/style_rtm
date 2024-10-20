# Copyright (c) OpenMMLab. All rights reserved.
# from .base_head import BaseHead
from .coord_cls_heads import RTMCCHead, SimCCHead, LightRTMCCHead, RTMCCHead2, RTMCCHead3, RTMCCHead4
# from .heatmap_heads import (AssociativeEmbeddingHead, CIDHead, CPMHead,
#                             HeatmapHead, MSPNHead, ViPNASHead)
# from .hybrid_heads import DEKRHead, VisPredictHead
# from .regression_heads import (DSNTHead, IntegralRegressionHead,
#                                RegressionHead, RLEHead, TemporalRegressionHead,
#                                TrajectoryRegressionHead)

# __all__ = [
#     'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
#     'RegressionHead', 'IntegralRegressionHead', 'SimCCHead', 'RLEHead',
#     'DSNTHead', 'AssociativeEmbeddingHead', 'DEKRHead', 'VisPredictHead',
#     'CIDHead', 'RTMCCHead', 'TemporalRegressionHead',
#     'TrajectoryRegressionHead'
# ]

from .heatmap_heads.custom_heads import CustomHead
from .heatmap_heads.u2head import U2Head
from .heatmap_heads.re2head import Re2Head
from .heatmap_heads.custom_base import CustomBaseHead
from .heatmap_heads.custom_base_v2 import CustomBaseHeadv2
from .heatmap_heads.custom_base256 import CustomBaseHead256
from .custom import PoseHead, PoseHead2, PoseHead3, StyleHead, PoseHead4, StyleRTMHead, StyleRTMHead2, StyleHeadExtra

__all__ = ['RTMCCHead', 'CustomHead', 'U2Head', 'Re2Head', 'CustomBaseHead', 
           'CustomBaseHeadv2', 'CustomBaseHead256', 'LightRTMCCHead', 'RTMCCHead2', 
           'RTMCCHead3', 'RTMCCHead4', 'PoseHead', 'PoseHead2', 'PoseHead3', 'StyleHead',
           'PoseHead4', 'StyleRTMHead', 'StyleRTMHead2', 'StyleHeadExtra']