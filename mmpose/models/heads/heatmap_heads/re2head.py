# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
from .custom_heads import U2NETP

OptIntSeq = Optional[Sequence[int]]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=17):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups)
        self.norm = nn.BatchNorm2d(num_features=self.groups)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=17):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups)
        self.norm = nn.BatchNorm2d(num_features=self.groups)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

@MODELS.register_module()
class Re2Head(BaseHead):
    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 hidden_channels: int = 256,
                 final_layer: dict = dict(kernel_size=1),
                 loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 u2net_weight: str = "/home/matsukawa/U-2-Net/saved_models/u2netp/u2netp.pth",
                 ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        self.cfg = dict(
                type='deconv',
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False)
        self.upsample1 = nn.Sequential(build_upsample_layer(cfg=self.cfg),
                                       nn.BatchNorm2d(num_features=self.hidden_channels),
                                       nn.ReLU(inplace=True))
        self.cfg = dict(
                type='deconv',
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False)
        self.upsample2 = nn.Sequential(build_upsample_layer(cfg=self.cfg),
                                       nn.BatchNorm2d(num_features=self.hidden_channels),
                                       nn.ReLU(inplace=True))
        
        self.res1 = nn.Sequential(nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=self.hidden_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=self.hidden_channels))
        self.res2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=self.hidden_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=self.hidden_channels))
        self.relu = nn.ReLU(inplace=True)
        
        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=self.hidden_channels,
                out_channels=self.out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # u2net
        self.u2net = U2NETP(3,1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.u2conv0 = nn.Sequential(nn.Conv2d(2, 1, 3, 1, 1),
                                    nn.Sigmoid())
        if u2net_weight:
            self.u2net.load_state_dict(torch.load(u2net_weight))
            for param in self.u2net.parameters():
                param.requires_grad = False


        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg=cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 17, 1))

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: tuple) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        # print("feats length: ", len(feats))
        # print(len(feats[0]), len(feats[1]))
        x = feats[0]
        so = feats[-1]
        so = self.u2net(so)
        so_avg = self.avgpool(so)
        so_max = self.maxpool(so)
        so = torch.cat((so_avg, so_max), dim=1)
        so = self.u2conv0(so)
        so0 = F.interpolate(so, scale_factor=1/2)
        so1 = F.interpolate(so, scale_factor=1/4)
        # so2 = F.interpolate(so, scale_factor=1/16)
        # print(type(x), type(so))
        # print("x, edge", x.shape, so.shape)

        x = self.upsample1(x)
        out = self.res1(x)
        x = x * so1 + out
        x = self.relu(x)

        x = self.upsample2(x)
        out = self.res2(x)
        x = x * so0 + out
        x = self.relu(x)

        x = self.final_layer(x)

        return x

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = self.forward(_feats_flip)
            _batch_heatmaps_flip = flip_heatmaps(
                _batch_heatmaps_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_heatmaps = self.forward(feats)

        preds = self.decode(batch_heatmaps)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        pred_fields = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss_kpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses.update(loss_kpt=loss_kpt)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`DeepposeRegressionHead` (before MMPose v1.0.0) to a
        compatible format of :class:`RegressionHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v
