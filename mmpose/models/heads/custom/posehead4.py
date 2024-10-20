import warnings
from typing import Tuple

import torch
from torchvision import models
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

@MODELS.register_module()
class PoseHead4(BaseHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 enc_cfg,
                 cls_out_size,
                 final_layer: dict = dict(kernel_size=1),
                 is_train = True, # False when testing
                 decoder = None,
                 loss_out = dict(type='KeypointMSELoss', use_target_weight=True),
                 init_cfg = None
                 ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # modules
        self.vgg = models.vgg16() if is_train else None
        if is_train:
            self.vgg.features = self.vgg.features[:7]
            self.vgg.avgpool = nn.AdaptiveAvgPool2d(output_size=cls_out_size)
            self.vgg.classifier[0] = nn.Linear(cls_out_size[0]*cls_out_size[1]*128, 512)
            self.vgg.classifier[3] = nn.Linear(512, 128)
            self.vgg.classifier[6] = nn.Linear(128, 21)

        # losses
        self.loss_output = MODELS.build(loss_out)
        self.loss_entropy = nn.CrossEntropyLoss()
        # decoder
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        self.pose_encoder = MODELS.build(enc_cfg)

        cfg = dict(
            type='Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)
        cfg.update(final_layer)
        self.final_layer = build_conv_layer(cfg)


    def forward(self, feats):
        x = feats[-1]
        x = self.pose_encoder(x)[-1]
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
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
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
        gt_class = torch.zeros([len(batch_data_samples), 21], device="cuda")
        ideal_class = torch.zeros([len(batch_data_samples), 21], device="cuda")
        for i in range(len(batch_data_samples)):
            num = int(batch_data_samples[i].id // 1e12)
            gt_class[i, num] = 1
            ideal_class[i, 0] = 1

        pred_fields = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        origin_class = self.vgg(feats[0])
        refine_class = self.vgg(feats[-1])

        # calculate losses
        losses = dict()
        loss = self.loss_output(pred_fields, gt_heatmaps, keypoint_weights)

        loss_img = self.loss_entropy(origin_class, gt_class) + self.loss_entropy(refine_class, ideal_class)

        losses.update(loss_kpt=loss, loss_img=loss_img)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def init_weights(self):
        super().init_weights()
        print("loads classifier weight.")
        weight = torch.load("/home/matsukawa_3/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
        weight = {k:v for k, v in weight.items() if "classifier" not in k}
        self.vgg.load_state_dict(weight, strict=False)