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


@MODELS.register_module()
class SamCP(DeformableDETR):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None,
                 bbox_roi_extractor: OptConfigType,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 encoder_patch: OptConfigType,
                 use_detach_menory=False,
                 as_two_stage=False,
                 use_image_encoder=False,
                 use_patch_query=False,
                 use_shared_mlp=False,
                 use_patch_encoder=False,
                 use_mask_roi=True,
                 use_mask_roi_mean=False,
                 cross_roi_patch=False,
                 use_denosing=False,
                 num_aug_rois=None,
                 with_box_refine=None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 **kwargs) -> None:
        self.as_two_stage = as_two_stage
        self.use_patch_query = use_patch_query
        self.num_aug_rois = num_aug_rois

        bbox_head['use_patch_query'] = use_patch_query
        if bbox_head['use_regression']:
            decoder['num_layers'] += 1
        super().__init__(*args, decoder=decoder, bbox_head=bbox_head, **kwargs)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        bbox_head['share_pred_layer'] = not with_box_refine
        bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
            if bbox_head['use_patch_cls'] or self.num_aug_rois or self.as_two_stage else decoder['num_layers']
        bbox_head['as_two_stage'] = as_two_stage
        self.bbox_head = MODELS.build(bbox_head)
        # assert self.as_two_stage, 'as_two_stage must be True for DINO'
        self.use_detach_menory = use_detach_menory
        self.use_patch_encoder = use_patch_encoder
        self.use_image_encoder = use_image_encoder
        self.use_denosing = use_denosing
        self.use_mask_roi = use_mask_roi
        self.use_mask_roi_mean = use_mask_roi_mean
        self.cross_roi_patch = cross_roi_patch
        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_things_classes if self.bbox_head.semantic_cfg is not None \
                else self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator_plus(**dn_cfg)
        self.encoder_patch = encoder_patch
        self.encoder_patch = DetrTransformerEncoder(**self.encoder_patch)

        self.num_shared_fcs = 2
        self.shared_fcs = nn.ModuleList()

        in_channels = bbox_roi_extractor['out_channels']
        self.in_channels = in_channels
        ft_size = bbox_roi_extractor['roi_layer']['output_size']

        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)

        self.use_shared_mlp = use_shared_mlp
        if self.use_shared_mlp:
            if self.num_aug_rois:
                out_channels = in_channels * self.num_aug_rois
            else:
                out_channels = in_channels
            self.shared_mlp = MLP(in_channels * ft_size * ft_size, 1024, out_channels, 2)
        else:
            for i in range(self.num_shared_fcs):
                self.shared_fcs.append(
                    nn.Linear(in_channels * ft_size * ft_size if i == 0 else in_channels, in_channels))
                # last_layer_dim = self.fc_out_channels
            self.relu = nn.ReLU(inplace=True)

            for p in self.shared_fcs.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
        self.roi_norm = nn.LayerNorm(self.embed_dims)
        ## patches position encoding
        prompt_embed_dim = int(in_channels / 2)
        self.num_point_embeddings: int = 2  # 2 box corners
        point_embeddings = [nn.Embedding(1, prompt_embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.pe_layer = PositionEmbeddingRandom(prompt_embed_dim // 2)
        self.ref_point_head = MLP(in_channels * 2, in_channels, in_channels, 2)
        if self.bbox_head.semantic_cfg is not None:
            self.semantic_query = MLP(self.bbox_head.semantic_text_embedding.shape[-1],
                                      in_channels, in_channels * 2, 2)
            self.semantic_fc = nn.Linear(self.bbox_head.semantic_text_embedding.shape[-1], self.embed_dims)
            self.semantic_norm = nn.LayerNorm(self.embed_dims)
            self.semantic_reference_fc = nn.Linear(self.bbox_head.semantic_text_embedding.shape[-1], 4)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = SamCPTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)
        else:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        if self.bbox_head.semantic_cfg is not None:
            self.query_embedding_semantic = nn.Embedding(self.bbox_head.num_stuff_classes,
                                                         self.embed_dims * 2)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        if not self.as_two_stage:
            if self.decoder.use_dab_querypos:
                self.reference_points_fc = nn.Linear(self.embed_dims, 4)
            else:
                self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def forward_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        patches_inputs_dict = self.pre_patches(img_feats, encoder_inputs_dict['valid_ratios'], batch_data_samples)
        if self.use_patch_encoder:
            patches_encoder_outputs_dict = self.forward_encoder_patch(**patches_inputs_dict)
            decoder_inputs_dict.update(patches_encoder_outputs_dict)
        else:
            decoder_inputs_dict.update(patches_inputs_dict)
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        if self.use_patch_encoder:
            encoder_outputs_dict.update(patches_encoder_outputs_dict)
        else:
            encoder_outputs_dict.update(patches_inputs_dict)
        use_unified_semantic = True
        if use_unified_semantic:
            pass
        tmp_dec_in, head_inputs_dict = self.pre_decoder(img_feats,
                                                        **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_inputs_dict.update(img_feats=img_feats, batch_data_samples=batch_data_samples)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict.update({'patch_fts': decoder_inputs_dict.get('patch_memory')})

        return head_inputs_dict

    def _embed_boxes(self, boxes: torch.Tensor, input_image_size) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[0].weight
        corner_embedding[:, 1, :] += self.point_embeddings[1].weight
        return corner_embedding

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
                mean_ft = (bbox_feats * proposals_segms[:, None]).flatten(2).sum(-1) / (
                        proposals_segms.flatten(1).sum(-1)[:, None] + 1e-10)
                bbox_feats = (bbox_feats * proposals_segms[:, None] +
                              mean_ft.detach()[..., None, None] * (1 - proposals_segms[:, None]))
            else:
                bbox_feats = bbox_feats * proposals_segms[:, None]
        if self.use_shared_mlp:
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
        if self.num_aug_rois:
            batch_bbox_masks = batch_bbox_masks[..., None].expand(*batch_bbox_masks.shape, self.num_aug_rois)
            b, pps, dim = batch_pos_embeds_patches.shape
            batch_pos_embeds_patches = batch_pos_embeds_patches.unsqueeze(2).expand(b, pps, self.num_aug_rois, dim)
            batch_proposals_boxes_ = batch_proposals_boxes_.unsqueeze(2).expand(b, pps, self.num_aug_rois, 4)

            batch_bbox_feats = rearrange(batch_bbox_feats, 'b pps rp dim -> b (pps rp) dim')
            batch_pos_embeds_patches = rearrange(batch_pos_embeds_patches, 'b pps rp dim -> b (pps rp) dim')
            batch_bbox_masks = rearrange(batch_bbox_masks, 'b pps rp -> b (pps rp)')
            batch_proposals_boxes_ = rearrange(batch_proposals_boxes_, 'b pps rp dim -> b (pps rp ) dim')

        patches_inputs_dict = dict(
            patch_memory=batch_bbox_feats,
            patch_memory_mask=batch_bbox_masks.bool(),
            patch_memory_pos=batch_pos_embeds_patches,
            patch_memory_coor=batch_proposals_boxes_)
        return patches_inputs_dict

    def forward_encoder_patch(self, patch_memory: Tensor, patch_memory_mask: Tensor,
                              patch_memory_pos: Tensor, patch_memory_coor: Tensor = None) -> Dict:
        memory = self.encoder_patch(
            query=patch_memory,
            query_pos=patch_memory_pos,
            key_padding_mask=patch_memory_mask,  # for self_attn
        )

        encoder_outputs_dict = dict(
            patch_memory=patch_memory if self.cross_roi_patch else memory,
            patch_memory_enc=memory,
            patch_memory_pos=patch_memory_pos,
            patch_memory_mask=patch_memory_mask,
            patch_memory_coor=patch_memory_coor)
        return encoder_outputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        if self.use_image_encoder:
            memory = self.encoder(
                query=feat,
                query_pos=feat_pos,
                key_padding_mask=feat_mask,  # for self_attn
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios)
            encoder_outputs_dict = dict(
                memory=memory,
                memory_pos=feat_pos,
                memory_mask=feat_mask,
                spatial_shapes=spatial_shapes)
        else:
            encoder_outputs_dict = dict(
                memory=feat,
                memory_pos=feat_pos,
                memory_mask=feat_mask,
                spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(
            self,
            mlvl_feats,
            memory: Tensor,
            memory_pos: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            patch_memory: Tensor = None,
            patch_memory_enc: Tensor = None,
            patch_memory_pos: Tensor = None,
            patch_memory_mask: Tensor = None,
            patch_memory_coor: Tensor = None,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:

        bs, _, c = memory.shape
        if self.as_two_stage:
            cls_out_features = self.bbox_head.cls_branches[
                self.decoder.num_layers].out_features

            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                          self.decoder.num_layers](output_memory) + output_proposals

            # NOTE The DINO selects top-k proposals according to scores of
            # multi-class classification, while DeformDETR, where the input
            # is `enc_outputs_class[..., 0]` selects according to scores of
            # binary classification.
            topk_indices = torch.topk(
                enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
            topk_score = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()

            query = self.query_embedding.weight[:, None, :]
            query = query.repeat(1, bs, 1).transpose(0, 1)
            if self.training:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples, num_matching_queries=len(query[0]))
                query = torch.cat([dn_label_query, query], dim=1)
                reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                             dim=1)
            else:
                reference_points = topk_coords_unact
                dn_mask, dn_meta = None, None
            reference_points = reference_points.sigmoid()

            decoder_inputs_dict = dict(
                query=query,
                query_pos=None,
                memory=memory,
                memory_pos=memory_pos,
                reference_points=reference_points,
                dn_mask=dn_mask)
            # NOTE DINO calculates encoder losses on scores and coordinates
            # of selected top-k encoder queries, while DeformDETR is of all
            # encoder queries.
            head_inputs_dict = dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta) if self.training else dict()
        elif self.use_patch_query:
            enc_outputs_class, enc_outputs_coord = None, None
            if self.use_detach_menory:
                output_memory = patch_memory_enc
                output_memory = output_memory.masked_fill(
                    patch_memory_mask.unsqueeze(-1), float(0))
                # output_memory = output_memory.masked_fill(~output_proposals_valid,
                #                                           float(0))
                output_memory = self.memory_trans_fc(output_memory)
                query = self.memory_trans_norm(output_memory).detach()
            elif self.num_aug_rois:

                enc_memory = patch_memory_enc.masked_fill(patch_memory_mask.unsqueeze(-1), float(0))
                enc_memory = self.memory_trans_fc(enc_memory)
                enc_memory = self.memory_trans_norm(enc_memory)
                outputs_feat = self.bbox_head.cls_branches[self.decoder.num_layers](enc_memory)
                if self.bbox_head.use_text_cls:
                    text_features = self.bbox_head.text_embedding.to(outputs_feat.device).float()[
                                    :self.bbox_head.num_classes]
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_feat_bias = torch.matmul(text_features, self.bbox_head.bias_lang) + self.bbox_head.bias0
                    text_features = self.bbox_head.text_porj(text_features / 0.1)  # (8, 256)
                    outputs_feat = outputs_feat / outputs_feat.norm(dim=-1, keepdim=True)
                    bias = text_feat_bias.unsqueeze(0).repeat(outputs_feat.shape[0], outputs_feat.shape[1], 1)

                    outputs_class = (outputs_feat @ text_features.t() / self.bbox_head.logit_scale.exp()) + bias
                    MAX_CLAMP_VALUE = 50000
                    outputs_class = torch.clamp(
                        outputs_class, max=MAX_CLAMP_VALUE)
                    outputs_class = torch.clamp(
                        outputs_class, min=-MAX_CLAMP_VALUE)
                else:
                    outputs_class = outputs_feat

                topk_indices = torch.topk(
                    outputs_class.max(-1)[0],
                    k=self.num_queries if outputs_class.shape[1] > self.num_queries else outputs_class.shape[1], dim=1)[
                    1]
                topk_score = torch.gather(
                    outputs_class, 1,
                    topk_indices.unsqueeze(-1).repeat(1, 1, self.bbox_head.num_classes))
                topk_feat = torch.gather(
                    enc_memory, 1,
                    topk_indices.unsqueeze(-1).repeat(1, 1, self.in_channels))
                topk_coor = torch.gather(
                    patch_memory_coor, 1,
                    topk_indices.unsqueeze(-1).repeat(1, 1, 4))
                patch_memory_mask = torch.gather(
                    patch_memory_mask, 1,
                    topk_indices)
                query_pos, query, query_coor = patch_memory_pos, topk_feat, topk_coor
            else:
                query_pos, query, query_coor = patch_memory_pos, patch_memory_enc, patch_memory_coor

            # img_shape = batch_data_samples[0].img_shape
            query_coor = bbox_xyxy_to_cxcywh(query_coor)

            bboxes_normalized_list = list()
            for sample, proposals_boxes in zip(batch_data_samples, query_coor):
                img_h, img_w = sample.img_shape
                factor = proposals_boxes.new_tensor([img_w, img_h, img_w,
                                                     img_h]).unsqueeze(0)
                bboxes_normalized = proposals_boxes / factor
                bboxes_normalized_list.append(bboxes_normalized)
            bboxes_normalized_list = torch.stack(bboxes_normalized_list)

            # pred_points[:, :, 0::2] = pred_points[:, :, 0::2] / img_shape[1]
            # pred_points[:, :, 1::2] = pred_points[:, :, 1::2] / img_shape[0]
            reference_points = inverse_sigmoid(bboxes_normalized_list)
            # query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query_mask = patch_memory_mask
            if self.bbox_head.semantic_cfg is not None:
                semantic_feat = self.bbox_head.semantic_text_embedding.to(query.device).float()
                semantic_query = self.semantic_fc(semantic_feat)
                semantic_query = self.semantic_norm(semantic_query)
                semantic_reference_points = self.semantic_reference_fc(semantic_feat)
                semantic_query = semantic_query.unsqueeze(0).expand(bs, -1, -1)
                semantic_reference_points = semantic_reference_points.unsqueeze(0).expand(bs, -1, -1)
                query_mask = torch.cat([query_mask.new_zeros(query.shape[0],
                                                             self.bbox_head.num_stuff_classes), query_mask], dim=1)

                sem_attn_mask = self.dn_query_generator.generate_dn_mask(
                    1 / 2, self.bbox_head.num_stuff_classes, device=query.device,
                    num_matching_queries=query.shape[1])
                query = torch.cat([semantic_query, query], dim=1)
                reference_points = torch.cat([semantic_reference_points, reference_points],
                                             dim=1)
            else:
                sem_attn_mask = None
            if self.use_denosing and self.training:
                ##########TODO###############
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples, num_matching_queries=len(query[0]))
                # if self.bbox_head.semantic_cfg is not None:
                #     dn_mask[-sem_attn_mask.shape[-1]:, -sem_attn_mask.shape[-1]:] = sem_attn_mask
                dino_use_roi = False
                if dino_use_roi:
                    dn_proposals = bbox_cxcywh_to_xyxy(dn_bbox_query.sigmoid() * factor)
                    rois = bbox2roi(dn_proposals)
                    batch_splits = [len(res) for res in dn_proposals]
                    dn_bbox_feats = self.bbox_roi_extractor(
                        mlvl_feats[:self.bbox_roi_extractor.num_inputs], rois)
                    if self.num_shared_fcs > 0:
                        dn_bbox_feats = dn_bbox_feats.flatten(1)
                        dn_bbox_feats = self.shared_mlp(dn_bbox_feats)
                        dn_bbox_feats = self.roi_norm(dn_bbox_feats)
                        # for fc in self.shared_fcs:
                        #     dn_bbox_feats = self.relu(fc(dn_bbox_feats))
                    dn_bbox_feats = dn_bbox_feats.reshape(dn_label_query.shape)
                    query = torch.cat([dn_bbox_feats, query], dim=1)
                else:
                    query = torch.cat([dn_label_query, query], dim=1)

                reference_points = torch.cat([dn_bbox_query, reference_points],
                                             dim=1)
                query_mask = torch.cat([query_mask.new_zeros(dn_bbox_query.shape[:2]), query_mask], dim=1)

                # dn_mask, dn_meta = None, None
            else:
                if self.bbox_head.semantic_cfg is not None:
                    dn_mask, dn_meta = None, None
                else:
                    dn_mask, dn_meta = None, None

            reference_points = reference_points.sigmoid()
            decoder_inputs_dict = dict(
                query=query,
                query_pos=None,
                query_mask=query_mask,
                memory=memory,
                memory_pos=memory_pos,
                reference_points=reference_points,
                sem_attn_mask=sem_attn_mask,
                dn_mask=dn_mask)
            head_inputs_dict = dict(
                enc_outputs_class=topk_score if self.num_aug_rois else None,
                enc_outputs_coord=bboxes_normalized_list if self.num_aug_rois else None,
                dn_meta=dn_meta) if self.training else dict()
        else:
            enc_outputs_class, enc_outputs_coord = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points_fc(query_pos)
            query_mask = query.new_zeros(query.shape[:2])
            if self.bbox_head.semantic_cfg is not None:
                semantic_feat = self.bbox_head.semantic_text_embedding.to(query.device).float()
                semantic_query = self.semantic_fc(semantic_feat)
                semantic_query = self.semantic_norm(semantic_query)
                semantic_reference_points = self.semantic_reference_fc(semantic_feat)
                semantic_query = semantic_query.unsqueeze(0).expand(bs, -1, -1)
                semantic_reference_points = semantic_reference_points.unsqueeze(0).expand(bs, -1, -1)
                query_mask = torch.cat([query_mask.new_zeros(query.shape[0],
                                                             self.bbox_head.num_stuff_classes), query_mask], dim=1)

                sem_attn_mask = self.dn_query_generator.generate_dn_mask(
                    1 / 2, self.bbox_head.num_stuff_classes, device=query.device,
                    num_matching_queries=query.shape[1])
                query = torch.cat([semantic_query, query], dim=1)
                reference_points = torch.cat([semantic_reference_points, reference_points],
                                             dim=1)
            else:
                sem_attn_mask = None
            if self.use_denosing and self.training:
                ##########TODO###############
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                    self.dn_query_generator(batch_data_samples, num_matching_queries=len(query[0]))
                # if self.bbox_head.semantic_cfg is not None:
                #     dn_mask[-sem_attn_mask.shape[-1]:, -sem_attn_mask.shape[-1]:] = sem_attn_mask
                dino_use_roi = False
                if dino_use_roi:
                    dn_proposals = bbox_cxcywh_to_xyxy(dn_bbox_query.sigmoid() * factor)
                    rois = bbox2roi(dn_proposals)
                    batch_splits = [len(res) for res in dn_proposals]
                    dn_bbox_feats = self.bbox_roi_extractor(
                        mlvl_feats[:self.bbox_roi_extractor.num_inputs], rois)
                    if self.num_shared_fcs > 0:
                        dn_bbox_feats = dn_bbox_feats.flatten(1)
                        for fc in self.shared_fcs:
                            dn_bbox_feats = self.relu(fc(dn_bbox_feats))
                    dn_bbox_feats = dn_bbox_feats.reshape(dn_label_query.shape)
                    query = torch.cat([dn_bbox_feats, query], dim=1)
                else:
                    query = torch.cat([dn_label_query, query], dim=1)

                reference_points = torch.cat([dn_bbox_query, reference_points],
                                             dim=1)
                query_mask = torch.cat([query_mask.new_zeros(dn_bbox_query.shape[:2]), query_mask], dim=1)

                # dn_mask, dn_meta = None, None
            else:
                if self.bbox_head.semantic_cfg is not None:
                    dn_mask, dn_meta = None, None
                else:
                    dn_mask, dn_meta = None, None
            reference_points = reference_points.sigmoid()
            decoder_inputs_dict = dict(
                query=query,
                query_pos=query_pos if not self.decoder.use_dab_querypos else None,
                query_mask=query_mask,
                memory=memory,
                memory_pos=memory_pos,
                reference_points=reference_points,
                sem_attn_mask=sem_attn_mask,
                dn_mask=dn_mask)
            # NOTE DINO calculates encoder losses on scores and coordinates
            # of selected top-k encoder queries, while DeformDETR is of all
            # encoder queries.
            head_inputs_dict = dict(
                enc_outputs_class=enc_outputs_class,
                enc_outputs_coord=enc_outputs_coord,
                dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        img_feats,
                        query: Tensor,
                        query_pos: Tensor,
                        query_mask,
                        memory: Tensor,
                        memory_pos: Tensor,
                        memory_mask: Tensor,
                        patch_memory: Tensor,
                        patch_memory_enc: Tensor,
                        patch_memory_mask: Tensor,
                        patch_memory_pos: Tensor,
                        patch_memory_coor: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        batch_data_samples,
                        sem_attn_mask,
                        dn_mask: Optional[Tensor] = None) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        if sem_attn_mask is None:
            if dn_mask is None:
                self_attn_mask = None
            else:
                self_attn_mask = dn_mask
        else:
            if dn_mask is None:
                self_attn_mask = sem_attn_mask
            else:
                self_attn_mask = dn_mask
                self_attn_mask[-sem_attn_mask.shape[-1]:, -sem_attn_mask.shape[-1]:] = sem_attn_mask

        inter_states, references, affinitys = self.decoder(
            query=query,
            query_pos=query_pos,
            query_mask=query_mask,
            value=memory,
            key_pos=memory_pos,
            key_padding_mask=memory_mask,
            patch_memory=rearrange(patch_memory, 'b (pps aug) d -> b pps aug d', aug=len(patch_memory))[:, :,
                         0] if self.num_aug_rois else patch_memory,
            # patch_memory_enc=rearrange(patch_memory_enc, 'b (pps aug) d -> b pps aug d', aug=len(patch_memory))[:, :,
            #              0] if self.num_aug_rois else patch_memory_enc,
            patch_memory_mask=rearrange(patch_memory_mask, 'b (pps aug)  -> b pps aug ', aug=len(patch_memory))[:, :,
                              0] if self.num_aug_rois else patch_memory_mask,
            patch_memory_pos=rearrange(patch_memory_pos, 'b (pps aug) d  -> b pps aug d', aug=len(patch_memory))[:, :,
                             0] if self.num_aug_rois else patch_memory_pos,
            patch_memory_coor=rearrange(patch_memory_coor, 'b (pps aug) d  -> b pps aug d', aug=len(patch_memory))[:, :,
                              0] if self.num_aug_rois else patch_memory_coor,
            self_attn_mask=self_attn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            batch_data_samples=batch_data_samples,
            reg_branches=self.bbox_head.reg_branches,
            bbox_roi_extractor=self.bbox_roi_extractor,
            roi_mlp=self.shared_mlp if self.use_shared_mlp else self.shared_fcs,
            roi_norm=self.roi_norm,
            img_feats=img_feats)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references, affinities=affinitys)
        return decoder_outputs_dict

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']
            if 'sem_results' in pred_results:
                data_sample.pred_sem_seg = pred_results['sem_results']
            # assert 'sem_results' not in pred_results, 'segmantic ' \
            #                                           'segmentation results are not supported yet.'

        return data_samples


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
