import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from ..base_head import BaseHead

@MODELS.register_module()
class PoseHead2(BaseHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_size,
                 in_featuremap_size,
                 cls_out_size,
                 simcc_split_ratio,
                 final_layer_kernel_size,
                 gau_cfg,
                 enc_cfg,
                 alpha = 0.1,
                 is_train = True, # False when testing
                 decoder = None,
                 loss_out = dict(type='KLDiscretLoss', use_target_weight=True),
                 init_cfg = None
                 ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.alpha = alpha

        # modules
        self.vgg = models.vgg16() if is_train else None
        if is_train:
            self.vgg.features = self.vgg.features[:7]
            self.vgg.avgpool = nn.AdaptiveAvgPool2d(output_size=cls_out_size)
            self.vgg.classifier[0] = nn.Linear(cls_out_size[0]*cls_out_size[1]*128, 512)
            self.vgg.classifier[3] = nn.Linear(512, 128)
            self.vgg.classifier[6] = nn.Linear(128, 20)

        # losses
        self.loss_output = MODELS.build(loss_out)
        self.loss_entropy = nn.CrossEntropyLoss()
        # decoder
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        self.pose_encoder = MODELS.build(enc_cfg)

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)


    def forward(self, feats):
        feat = feats[-1]
        
        pred = self.pose_encoder(feat)[-1]

        pred = self.final_layer(pred)
        pred = torch.flatten(pred, 2)
        pred = self.mlp(pred)
        pred = self.gau(pred)

        pred_x = self.cls_x(pred)
        pred_y = self.cls_y(pred)

        return pred_x, pred_y

    def predict(self, 
                feats: Tuple[torch.Tensor], 
                batch_data_samples,
                test_cfg):
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)

            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):

                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    def loss(self, 
             feats, 
             batch_data_samples, 
             train_cfg):
        L = len(batch_data_samples)
        gt_class = torch.zeros([L, 20], device="cuda")
        ideal_class = torch.full((L, 20), fill_value=1.0/L, device="cuda")
        for i in range(L):
            num = int(batch_data_samples[i].id // 1e12) - 1
            gt_class[i, num] = 1

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_x, pred_y = self.forward(feats)
        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        origin_class = self.vgg(feats[0])
        refine_class = self.vgg(feats[-1])

        # calculate losses
        losses = dict()
        loss = self.loss_output(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        loss_img = (self.loss_entropy(origin_class, gt_class) + self.loss_entropy(refine_class, ideal_class)) * self.alpha
        losses.update(loss_img=loss_img)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    def init_weights(self):
        super().init_weights()
        print("loads classifier weight.")
        weight = torch.load("/home/matsukawa_3/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
        weight = {k:v for k, v in weight.items() if "classifier" not in k}
        self.vgg.load_state_dict(weight, strict=False)