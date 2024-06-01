# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple, List

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
from mmdet.structures import OptSampleList, SampleList
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.models.layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder, DetrTransformerEncoder,
                                 DinoTransformerDecoder, SinePositionalEncoding)
from mmdet.models.detectors.dino import DINO
from mmdet.models.detectors.deformable_detr import DeformableDETR
from mmdet.structures.bbox import bbox2roi, bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from typing import Any, Optional, Tuple, Type, Union
import numpy as np
from mmdet.models.detectors.deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from ..layers.transformer.samcp_layers import SamCPTransformerDecoder
from mmdet.models.layers.transformer.utils import coordinate_to_encoding, MLP
from mmdet.models._SAM_CP.layers.transformer.samcp_layers import CdnQueryGenerator_plus, SamCPTransformerEncoder
from einops import rearrange, repeat
from .SamCP import SamCP

@MODELS.register_module()
class OpenSamCP(SamCP):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def pre_patches(
            self, mlvl_feats: Tuple[Tensor], valid_ratios: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None

        proposals_bboxes = [data_samples.proposals.bboxes for data_samples in batch_data_samples]

        rois = bbox2roi(proposals_bboxes)
        batch_splits = [len(res) for res in proposals_bboxes]
        bbox_feats = self.bbox_roi_extractor(
            mlvl_feats[:self.bbox_roi_extractor.num_inputs], rois)

        if self.use_mask_roi:
            proposals_segms = [data_samples.proposals.roi_segms for data_samples in batch_data_samples]
            proposals_segms = torch.cat([bbox_feats.new_tensor(mm.masks) for mm in proposals_segms])
            if self.use_mask_roi_mean:
                mean_ft=(bbox_feats * proposals_segms[:, None]).flatten(2).sum(-1)/(proposals_segms.flatten(1).sum(-1)[:,None]+1e-10)
                bbox_feats = (bbox_feats * proposals_segms[:, None]+
                              mean_ft.detach()[...,None,None]*(1-proposals_segms[:, None]))
            else:
                bbox_feats = bbox_feats * proposals_segms[:, None]
        if self.use_clip_attnpool:
            bbox_feats=self.backbone.attnpool(bbox_feats)

        elif self.use_shared_mlp:
            bbox_feats = self.shared_mlp(bbox_feats.flatten(1))
        else:
            if self.num_shared_fcs > 0:
                bbox_feats = bbox_feats.flatten(1)
                for fc in self.shared_fcs:
                    bbox_feats = self.relu(fc(bbox_feats))

        if self.num_aug_rois:
            bbox_feats = bbox_feats.reshape(bbox_feats.shape[0], self.num_aug_rois, -1)
        bbox_feats = self.roi_norm(bbox_feats)
        batch_bbox_feats = torch.split(bbox_feats, batch_splits)
        max_patches_dim = max(batch_splits)
        batch_bbox_feats = [torch.cat([ft, ft.new_zeros(max_patches_dim - len(ft), *ft.shape[1:])])
                            for ft in batch_bbox_feats]

        batch_bbox_feats = torch.stack(batch_bbox_feats)

        ### attn mask
        batch_bbox_masks = bbox_feats.new_ones((batch_size, max_patches_dim), dtype=torch.long)
        for img_id in range(batch_size):
            batch_bbox_masks[img_id, :batch_splits[img_id]] = 0

        ### pos embedding
        batch_proposals_boxes = [torch.cat([ft.tensor, ft.new_zeros(max_patches_dim - len(ft), 4)])
                                 for ft in proposals_bboxes]
        batch_proposals_boxes = torch.stack(batch_proposals_boxes)
        batch_proposals_boxes_ = copy.deepcopy(batch_proposals_boxes)
        batch_proposals_boxes = bbox_xyxy_to_cxcywh(batch_proposals_boxes)
        bboxes_normalized_list = list()
        for sample, proposals_boxes in zip(batch_data_samples, batch_proposals_boxes):
            img_h, img_w = sample.img_shape
            factor = proposals_boxes.new_tensor([img_w, img_h, img_w,
                                                 img_h]).unsqueeze(0)
            bboxes_normalized = proposals_boxes / factor
            bboxes_normalized_list.append(bboxes_normalized)
        bboxes_normalized_list = torch.stack(bboxes_normalized_list)

        batch_proposals_boxes = bboxes_normalized_list[:, :, None] * torch.cat(
            [valid_ratios, valid_ratios], -1)[:, None]
        batch_proposals_boxes = batch_proposals_boxes[:, :, 0, :]

        batch_pos_embeds_patches = self.ref_point_head(
            coordinate_to_encoding(batch_proposals_boxes, temperature=self.positional_encoding.temperature))
        # self._embed_boxes(batch_proposals_boxes, batch_data_samples[0].img_shape)


        patches_inputs_dict = dict(
            patch_memory=batch_bbox_feats,
            patch_memory_mask=batch_bbox_masks.bool(),
            patch_memory_pos=batch_pos_embeds_patches,
            patch_memory_coor=batch_proposals_boxes_)
        return patches_inputs_dict

