from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                 Predictions)
from ..base_head import BaseHead

class GroupPose(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False,
                    num_feature_levels=1,
                    nheads=8,
                    two_stage_type='no',
                    dec_pred_class_embed_share=False,
                    dec_pred_pose_embed_share=False,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    cls_no_bias = False,
                    num_body_points = 17
                    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.num_body_points = num_body_points      

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss

        # prepare class
        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_point_embed.layers[-1].bias.data, 0)
        
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_pose_embed_share:
            pose_embed_layerlist = [_point_embed for i in range(transformer.num_decoder_layers)]
        else:
            pose_embed_layerlist = [copy.deepcopy(_point_embed) for i in range(transformer.num_decoder_layers)]

        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.num_body_points = num_body_points

        # two stage
        _keypoint_embed = MLP(hidden_dim, 2*hidden_dim, 2*num_body_points, 4)
        nn.init.constant_(_keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_keypoint_embed.layers[-1].bias.data, 0)
        
        if two_stage_bbox_embed_share:
            self.transformer.enc_pose_embed = _keypoint_embed
        else:
            self.transformer.enc_pose_embed = copy.deepcopy(_keypoint_embed)

        if two_stage_class_embed_share:
            self.transformer.enc_out_class_embed = _class_embed
        else:
            self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)


        hs_pose, refpoint_pose, mix_refpoint, mix_embedding = self.transformer(srcs, masks, poss)
        
        outputs_class=[]
        outputs_keypoints_list = []
        
        for dec_lid, (hs_pose_i, refpoint_pose_i, layer_pose_embed, layer_cls_embed) in enumerate(zip(hs_pose, refpoint_pose, self.pose_embed, self.class_embed)):
            # pose
            bs, nq, np = refpoint_pose_i.shape
            refpoint_pose_i = refpoint_pose_i.reshape(bs, nq, np // 2, 2)
            delta_pose_unsig = layer_pose_embed(hs_pose_i[:, :, 1:])
            layer_outputs_pose_unsig = inverse_sigmoid(refpoint_pose_i[:, :, 1:]) + delta_pose_unsig
            vis_flag = torch.ones_like(layer_outputs_pose_unsig[..., -1:], device=layer_outputs_pose_unsig.device)
            layer_outputs_pose_unsig = torch.cat([layer_outputs_pose_unsig, vis_flag], dim=-1).flatten(-2)
            layer_outputs_pose_unsig = layer_outputs_pose_unsig.sigmoid()
            outputs_keypoints_list.append(keypoint_xyzxyz_to_xyxyzz(layer_outputs_pose_unsig))
            
            # cls
            layer_cls = layer_cls_embed(hs_pose_i[:, :, 0])
            outputs_class.append(layer_cls)

        out = {'pred_logits': outputs_class[-1], 'pred_keypoints': outputs_keypoints_list[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_keypoints_list)

        # for encoder output
        if mix_refpoint is not None and mix_embedding is not None:
            # prepare intermediate outputs
            interm_class = self.transformer.enc_out_class_embed(mix_embedding)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_keypoints': mix_refpoint}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_keypoints):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_keypoints': c}
                for a, c in zip(outputs_class[:-1], outputs_keypoints[:-1])]

@MODELS.register_module()
class custom_GroupPose(BaseHead):
    def __init__(
                self, 
                in_channels, 
                num_joints, 
                loss, 
                decoder, 
                init_cfg
                ):
        super().__init__(init_cfg)

        # variable
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINTS_CODECS.build(decoder)
        else:
            self.decoder = None

        # Head Blocks


    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        x = feats[-1]

        return x.reshape(-1, self.num_joints, 2)

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords = self.forward(_feats)
            _batch_coords_flip = flip_coordinates(
                self.forward(_feats_flip),
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        return preds

    def loss(
            self,
            inputs,
            batch_data_samples,
            train_cfg
            ):
        pred_outputs = self.forward(inputs)

        keypoint_labels = torch.cat(
            [d.gt_instance_labels.keypoint_labels for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_outputs, keypoint_labels,
                                keypoint_weights.unsqueeze(-1))

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_outputs),
            gt=to_numpy(keypoint_labels),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_outputs.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=keypoint_labels.device)
        losses.update(acc_pose=acc_pose)

        return losses
