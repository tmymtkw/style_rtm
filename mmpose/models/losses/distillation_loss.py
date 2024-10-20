import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS

@MODELS.register_module()
class DistillationLoss(nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.weight = weight

    def forward(self, feat_s, feat_t):
        loss = self.criterion(feat_s, feat_t)
        return loss * self.weight