# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple, Union
import math
import torch
from mmengine.model import BaseModule
from torch import Tensor, nn
from mmengine import ConfigDict
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import OptConfigType
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerDecoder
from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding, inverse_sigmoid
from mmengine.model import ModuleList

from mmdet.models.layers.transformer.detr_layers import DetrTransformerDecoderLayer
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.utils import ConfigType
from mmdet.structures.bbox import bbox2roi


class SamCPTransformerEncoder(BaseModule):
    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            SamCPTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        cross_attn_mask = None
        for layer in self.layers:
            query, affinity = layer(query, query_pos, key_padding_mask, cross_attn_mask, **kwargs)

            aff = affinity.sigmoid()
            max = aff.masked_fill(key_padding_mask[:, None], 0).max(-1)[0][:, :, None]
            min = aff.masked_fill(key_padding_mask[:, None], 1).min(-1)[0][:, :, None]
            normed_affinity = (aff - min) / (max - min)
            cross_attn_mask = normed_affinity < 0.5
        return query


class SamCPTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer decoder of DINO."""

    def __init__(self, *args, use_dab_querypos=False, pos_temperature=10000, use_affinity_refine=False,
                 share_affinity=False, enhance_query=False, **kwargs):
        self.use_dab_querypos = use_dab_querypos
        self.pos_temperature = pos_temperature
        self.use_affinity_refine = use_affinity_refine
        self.share_affinity = share_affinity
        self.enhance_query = enhance_query

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""

        self.layers = ModuleList([
            SamInstTransformerDecoderLayer(**self.layer_cfg, share_affinity=self.share_affinity)
            for _ in range(self.num_layers)
        ])
        self.calculate_affinity = Calculate_affinity() if self.share_affinity else None
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
        self.roi_norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, query_pos: Tensor, query_mask, value: Tensor, key_pos: Tensor,
                key_padding_mask: Tensor,
                patch_memory: Tensor, patch_memory_mask: Tensor, patch_memory_pos: Tensor, patch_memory_coor: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, batch_data_samples, reg_branches: nn.ModuleList,
                bbox_roi_extractor, roi_mlp, roi_norm, img_feats,
                **kwargs) -> Tuple[Tensor]:
        intermediate = []
        intermediate_reference_points = [reference_points]
        # intermediate_reference_points = []
        intermediate_affinity = []
        cross_attn_mask = None
        affinity = None
        affinity_refine = None
        for lid, layer in enumerate(self.layers):
            if self.use_dab_querypos and affinity is not None:
                if pred_points.shape[-1] == 4:
                    pred_points_input = \
                        pred_points[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None]
                else:
                    assert reference_points.shape[-1] == 2
                    pred_points_input = \
                        pred_points[:, :, None] * valid_ratios[:, None]
                query_sine_embed = coordinate_to_encoding(
                    pred_points_input[:, :, 0, :], temperature=self.pos_temperature)
                query_pos = self.ref_point_head(query_sine_embed)
            else:
                if reference_points.shape[-1] == 4:
                    reference_points_input = \
                        reference_points[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None]
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = \
                        reference_points[:, :, None] * valid_ratios[:, None]

                if query_pos is None:
                    query_sine_embed = coordinate_to_encoding(
                        reference_points_input[:, :, 0, :], temperature=self.pos_temperature)
                    query_pos = self.ref_point_head(query_sine_embed)

            tmp = affinity if affinity is not None else 0
            query, affinity = layer(
                query,
                query_pos=query_pos,
                query_mask=query_mask,
                value=value,
                key=value,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                patch_memory=patch_memory,
                patch_memory_mask=patch_memory_mask,
                patch_memory_pos=patch_memory_pos,
                patch_memory_coor=patch_memory_coor,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                cross_attn_mask=cross_attn_mask,
                affinity_brench=self.calculate_affinity,
                **kwargs)
            if self.use_affinity_refine:
                affinity = affinity + tmp

            # if reg_branches is not None:
            #     tmp = reg_branches[lid](query)
            #     assert reference_points.shape[-1] == 4
            #     new_reference_points = tmp + inverse_sigmoid(
            #         reference_points, eps=1e-3)
            #     new_reference_points = new_reference_points.sigmoid()
            #     reference_points = new_reference_points.detach()
            aff = affinity.sigmoid()
            max = aff.masked_fill(patch_memory_mask[:, None], 0).max(-1)[0][:, :, None]
            min = aff.masked_fill(patch_memory_mask[:, None], 1).min(-1)[0][:, :, None]
            normed_affinity = (aff - min) / (max - min)
            cross_attn_mask = normed_affinity

            pred_points = get_bboxes_segms(affinity, patch_memory_coor, patch_memory_mask,
                                           batch_data_samples[0].batch_input_shape, True, 0.5).detach()

            if self.enhance_query:
                rois = bbox2roi(pred_points)
                bbox_feats = bbox_roi_extractor(
                    img_feats[:bbox_roi_extractor.num_inputs], rois)
                roi_feat = roi_mlp(bbox_feats.flatten(1)).reshape(query.shape)
                query = (roi_norm(roi_feat)+query)/2
            query_out = query

            pred_points = bbox_xyxy_to_cxcywh(pred_points)
            # pred_points[:, :, 0::2] = pred_points[:, :, 0::2] / img_shape[1]
            # pred_points[:, :, 1::2] = pred_points[:, :, 1::2] / img_shape[0]
            bboxes_normalized_list = list()
            for sample, proposals_boxes in zip(batch_data_samples, pred_points):
                img_h, img_w = sample.img_shape
                factor = proposals_boxes.new_tensor([img_w, img_h, img_w,
                                                     img_h]).unsqueeze(0)
                bboxes_normalized = proposals_boxes / factor
                bboxes_normalized_list.append(bboxes_normalized)
            pred_points = torch.stack(bboxes_normalized_list)

            if self.return_intermediate:
                intermediate.append(self.norm(query_out))
                intermediate_reference_points.append(
                    pred_points.detach())
                intermediate_affinity.append(affinity)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), torch.stack(intermediate_affinity)

        return query_out, reference_points, affinity




class SamCPTransformerEncoderLayer(BaseModule):

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 use_deformable=False,
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:
        self.use_deformable = use_deformable
        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
        self.fc_affinity_proposals = FFN(**self.ffn_cfg)
        self.fc_affinity_patches = FFN(**self.ffn_cfg)
        self.logit_scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(self.embed_dims), requires_grad=True)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
        self.conv_aff = nn.ModuleList()
        num_head = 8
        self.conv_aff.append(nn.Conv2d(num_head, num_head, kernel_size=1))
        self.conv_aff.append(nn.Conv2d(num_head, num_head, kernel_size=1))
        self.conv_aff.append(nn.Conv2d(num_head, 1, kernel_size=1))
        self.relu = nn.ReLU(inplace=True)
        for p in self.conv_aff.parameters():  ##init the weight
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, cross_attn_mask=None, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        patch_memory = query
        if cross_attn_mask is not None:
            if cross_attn_mask.dim() == 3:
                cross_attn_mask = cross_attn_mask[:, None].repeat(1, self.self_attn.num_heads, 1, 1)
                cross_attn_mask = cross_attn_mask.reshape(-1, *cross_attn_mask.shape[-2:])

        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            attn_mask=cross_attn_mask if self.use_deformable else None,
            **kwargs)

        affinity = self.calculate_affinity(query + query_pos, patch_memory + query_pos)

        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query, affinity

    def calculate_affinity(self, affinity_proposals, affinity_patches, num_heads=8):
        feat_bias = torch.matmul(affinity_patches, self.bias_lang) + self.bias0
        affinity_proposals = self.fc_affinity_proposals(affinity_proposals)
        affinity_patches = self.fc_affinity_patches(affinity_patches)

        affinity_proposals, affinity_patches = affinity_proposals.permute(1, 0, 2), affinity_patches.permute(1, 0, 2)
        tgt_len, bsz, embed_dim = affinity_proposals.shape
        src_len, _, _ = affinity_patches.shape

        head_dim = self.embed_dims // num_heads
        affinity_proposals = affinity_proposals.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        affinity_patches = affinity_patches.contiguous().reshape(affinity_patches.shape[0], bsz * num_heads,
                                                                 head_dim).transpose(0, 1)
        # v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        B, Nt, E = affinity_proposals.shape
        affinity_proposals = affinity_proposals / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn_affinity = torch.bmm(affinity_proposals, affinity_patches.transpose(-2, -1))
        attn_affinity = attn_affinity.view(bsz, num_heads, tgt_len, src_len)
        for conv in self.conv_aff:
            attn_affinity = conv(self.relu(attn_affinity))
        # attn_affinity = attn_affinity.sum(dim=1)/num_heads
        bias = feat_bias.unsqueeze(1).repeat(1, affinity_proposals.shape[1], 1)
        attn_affinity = attn_affinity.squeeze(1) / self.logit_scale.exp() + bias

        return attn_affinity


class SamInstTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def __init__(self, *args, use_patch_cross=True, query_cross_patch=True, use_deformable=False,
                 use_query_pos_affinity=False, soft_deformable=False, share_affinity=False, use_qeury_aff=False,
                 **kwargs):
        self.use_patch_cross = use_patch_cross
        self.query_cross_patch = query_cross_patch
        self.use_deformable = use_deformable
        self.use_query_pos_affinity = use_query_pos_affinity
        self.share_affinity = share_affinity
        self.use_qeury_aff = use_qeury_aff
        self.soft_deformable = soft_deformable
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        if self.use_qeury_aff:
            self.self_attn_aff = MultiheadAttention(**self.self_attn_cfg)
        self.self_attn_patch = MultiheadAttention(**self.self_attn_cfg)
        if self.query_cross_patch:
            self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        else:
            self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.cross_path_attn = MultiheadAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(5)
        ]
        self.norms = ModuleList(norms_list)
        if not self.share_affinity:
            self.fc_affinity_proposals = FFN(**self.ffn_cfg)
            self.fc_affinity_patches = FFN(**self.ffn_cfg)

            self.logit_scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
            self.bias_lang = nn.Parameter(torch.zeros(self.embed_dims), requires_grad=True)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
            self.conv_aff = nn.ModuleList()
            num_head = 8
            self.conv_aff.append(nn.Conv2d(num_head, num_head, kernel_size=1))
            self.conv_aff.append(nn.Conv2d(num_head, num_head, kernel_size=1))
            self.conv_aff.append(nn.Conv2d(num_head, 1, kernel_size=1))
            self.relu = nn.ReLU(inplace=True)
            for p in self.conv_aff.parameters():  ##init the weight
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)

    def forward(self,
                query: Tensor,
                query_mask=None,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                patch_memory: Tensor = None,
                patch_memory_mask: Tensor = None,
                patch_memory_pos: Tensor = None,
                patch_memory_coor: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                affinity_brench=None,
                **kwargs) -> Tensor:

        # query = self.cross_attn(
        #     query=query,
        #     key=patch_memory,
        #     value=patch_memory,
        #     query_pos=query_pos,
        #     key_pos=patch_memory_pos,
        #     attn_mask=cross_attn_mask,
        #     key_padding_mask=patch_memory_mask,
        #     **kwargs)

        # query = self.cross_attn(
        #     query=query,
        #     key=key,
        #     value=value,
        #     query_pos=query_pos,
        #     key_pos=key_pos,
        #     attn_mask=cross_attn_mask,
        #     key_padding_mask=key_padding_mask,
        #     **kwargs)
        if self.use_qeury_aff:
            query_aff = query
            query_aff = self.self_attn_aff(
                query=query_aff,
                key=query,
                value=query,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=self_attn_mask,
                **kwargs
            )
            query_aff = self.norms[4](query_aff)
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=query_mask,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[1](query)

        if self.use_patch_cross:
            patch_memory = self.cross_path_attn(
                query=patch_memory,
                key=query,
                value=query,
                query_pos=patch_memory_pos,
                key_pos=query_pos,
                attn_mask=None,
                key_padding_mask=None,
            )
            patch_memory = self.norms[3](patch_memory)
        if self.use_query_pos_affinity:
            if self.share_affinity:
                affinity = affinity_brench(query + query_pos, patch_memory + patch_memory_pos)
            else:
                if self.use_qeury_aff:
                    affinity = self.calculate_affinity(query_aff + query_pos, patch_memory + patch_memory_pos)
                else:
                    affinity = self.calculate_affinity(query + query_pos, patch_memory + patch_memory_pos)
        else:
            if self.share_affinity:
                affinity = affinity_brench(query, patch_memory + patch_memory_pos)
            else:
                affinity = self.calculate_affinity(query, patch_memory + patch_memory_pos)

        if self.query_cross_patch:
            if cross_attn_mask is not None:
                if cross_attn_mask.dim() == 3:
                    if not self.soft_deformable:
                        cross_attn_mask = cross_attn_mask < 0.5
                    cross_attn_mask = cross_attn_mask[:, None].repeat(1, self.cross_attn.num_heads, 1, 1)
                    cross_attn_mask = cross_attn_mask.reshape(-1, *cross_attn_mask.shape[-2:])
            query = self.cross_attn(
                query=query,
                key=patch_memory,
                value=patch_memory,
                query_pos=query_pos,
                key_pos=patch_memory_pos,
                attn_mask=cross_attn_mask if self.use_deformable else None,
                key_padding_mask=patch_memory_mask,
                **kwargs)
        else:
            query = self.cross_attn(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        query = self.norms[0](query)

        query = self.ffn(query)
        query = self.norms[2](query)

        return query, affinity

    def calculate_affinity(self, affinity_proposals, affinity_patches, num_heads=8):
        feat_bias = torch.matmul(affinity_patches, self.bias_lang) + self.bias0
        affinity_proposals = self.fc_affinity_proposals(affinity_proposals)
        affinity_patches = self.fc_affinity_patches(affinity_patches)

        affinity_proposals, affinity_patches = affinity_proposals.permute(1, 0, 2), affinity_patches.permute(1, 0, 2)
        tgt_len, bsz, embed_dim = affinity_proposals.shape
        src_len, _, _ = affinity_patches.shape

        head_dim = self.embed_dims // num_heads
        affinity_proposals = affinity_proposals.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        affinity_patches = affinity_patches.contiguous().reshape(affinity_patches.shape[0], bsz * num_heads,
                                                                 head_dim).transpose(0, 1)
        # v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        B, Nt, E = affinity_proposals.shape
        affinity_proposals = affinity_proposals / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn_affinity = torch.bmm(affinity_proposals, affinity_patches.transpose(-2, -1))
        attn_affinity = attn_affinity.view(bsz, num_heads, tgt_len, src_len)
        for conv in self.conv_aff:
            attn_affinity = conv(self.relu(attn_affinity))
        # attn_affinity = attn_affinity.sum(dim=1)/num_heads
        bias = feat_bias.unsqueeze(1).repeat(1, affinity_proposals.shape[1], 1)
        attn_affinity = attn_affinity.squeeze(1) / self.logit_scale.exp() + bias

        return attn_affinity

    def calculate_affinity2(self, affinity_proposals, affinity_patches):
        feat_bias = torch.matmul(affinity_patches, self.bias_lang) + self.bias0
        affinity_proposals = self.fc_affinity_proposals(affinity_proposals)
        affinity_patches = self.fc_affinity_patches(affinity_patches)
        affinity_proposals = affinity_proposals / affinity_proposals.norm(dim=-1, keepdim=True)
        affinity_patches = affinity_patches / affinity_patches.norm(dim=-1, keepdim=True)
        # affinity_patches=

        bias = feat_bias.unsqueeze(1).repeat(1, affinity_proposals.shape[1], 1)
        attn_affinity = (torch.bmm(affinity_proposals,
                                   affinity_patches.transpose(-2, -1)) / self.logit_scale.exp()) + bias

        return attn_affinity


class Calculate_affinity(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 init_cfg: Union[dict, ConfigDict] = None, ):
        super().__init__(init_cfg)
        self.ffn_cfg = ffn_cfg
        self.embed_dims = embed_dims
        self._init_layers()

    def _init_layers(self) -> None:
        self.fc_affinity_proposals = FFN(**self.ffn_cfg)
        self.fc_affinity_patches = FFN(**self.ffn_cfg)

        self.logit_scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(self.embed_dims), requires_grad=True)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
        self.conv_aff = nn.ModuleList()
        num_head = 8
        self.conv_aff.append(nn.Conv2d(num_head, num_head, kernel_size=1))
        self.conv_aff.append(nn.Conv2d(num_head, num_head, kernel_size=1))
        self.conv_aff.append(nn.Conv2d(num_head, 1, kernel_size=1))
        self.relu = nn.ReLU(inplace=True)
        for p in self.conv_aff.parameters():  ##init the weight
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, affinity_proposals, affinity_patches, num_heads=8):
        feat_bias = torch.matmul(affinity_patches, self.bias_lang) + self.bias0
        affinity_proposals = self.fc_affinity_proposals(affinity_proposals)
        affinity_patches = self.fc_affinity_patches(affinity_patches)

        affinity_proposals, affinity_patches = affinity_proposals.permute(1, 0, 2), affinity_patches.permute(1, 0, 2)
        tgt_len, bsz, embed_dim = affinity_proposals.shape
        src_len, _, _ = affinity_patches.shape

        head_dim = self.embed_dims // num_heads
        affinity_proposals = affinity_proposals.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        affinity_patches = affinity_patches.contiguous().reshape(affinity_patches.shape[0], bsz * num_heads,
                                                                 head_dim).transpose(0, 1)
        # v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        B, Nt, E = affinity_proposals.shape
        affinity_proposals = affinity_proposals / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn_affinity = torch.bmm(affinity_proposals, affinity_patches.transpose(-2, -1))
        attn_affinity = attn_affinity.view(bsz, num_heads, tgt_len, src_len)
        for conv in self.conv_aff:
            attn_affinity = conv(self.relu(attn_affinity))
        # attn_affinity = attn_affinity.sum(dim=1)/num_heads
        bias = feat_bias.unsqueeze(1).repeat(1, affinity_proposals.shape[1], 1)
        attn_affinity = attn_affinity.squeeze(1) / self.logit_scale.exp() + bias

        return attn_affinity


def get_bboxes_segms(affinity_pred, proposals, patch_memory_mask, img_shape, norm=False, thre=0.5):
    # proposal_boxes = proposals.bboxes
    affinity_pred = affinity_pred.sigmoid()
    if norm:
        max = affinity_pred.masked_fill(patch_memory_mask[:, None], 0).max(-1)[0][:, :, None]
        min = affinity_pred.masked_fill(patch_memory_mask[:, None], 1).min(-1)[0][:, :, None]
        affinity_pred = (affinity_pred - min) / (max - min)
        affinity_pred.masked_fill(patch_memory_mask[:, None], 0)

    aa = ((affinity_pred[..., None] > thre) * proposals[:, None, ...])
    ignorezero = (affinity_pred > thre) == 0
    aa[ignorezero, :] = affinity_pred.new([float('inf'), float('inf'), float('-inf'), float('-inf')])
    x1y1 = aa[..., :2].min(2)[0]
    x2y2 = aa[..., 2:].max(2)[0]
    bboxes_pred = torch.cat([x1y1, x2y2], dim=-1)
    ignorezero_b = (affinity_pred > thre).max(-1)[0] == 0
    bboxes_pred[ignorezero_b] = affinity_pred.new([float(0), float(0), float(img_shape[1]), float(img_shape[0])])
    return bboxes_pred


class CdnQueryGenerator_plus(BaseModule):

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 use_text_as_label=False,
                 category_embeddings='pretrained/clip/clip_categories_embedding/coco_80_with_bg_vit-b.pth',
                 group_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        # prepare grouping strategy
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        if self.dynamic_dn_groups:
            if 'num_dn_queries' not in group_cfg:
                warnings.warn("'num_dn_queries' should be set when using "
                              'dynamic dn groups, use 100 as default.')
            self.num_dn_queries = group_cfg.get('num_dn_queries', 100)
            assert isinstance(self.num_dn_queries, int), \
                f'Expected the num_dn_queries to have type int, but got ' \
                f'{self.num_dn_queries}({type(self.num_dn_queries)}). '
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using static dn groups'
            self.num_groups = group_cfg['num_groups']
            assert isinstance(self.num_groups, int), \
                f'Expected the num_groups to have type int, but got ' \
                f'{self.num_groups}({type(self.num_groups)}). '

        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates `Unknown` class. However, the embedding of `unknown` class
        # is not used in the original DINO.
        # TODO: num_classes + 1 or num_classes ?
        self.use_text_as_label = use_text_as_label
        if self.use_text_as_label:
            self.text_embedding = torch.load(category_embeddings, map_location=torch.device('cpu'))
            self.label_mlp = MLP(self.text_embedding.shape[-1],
                                 self.embed_dims, self.embed_dims, 2)
        else:
            self.label_embedding = nn.Embedding(self.num_classes, self.embed_dims)

    def __call__(self, batch_data_samples: SampleList, num_matching_queries) -> tuple:

        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            gt_labels_list.append(sample.gt_instances.labels)
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device, num_matching_queries=num_matching_queries)

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups)

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def get_num_groups(self, max_num_target: int = None) -> int:

        if self.dynamic_dn_groups:
            assert max_num_target is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if max_num_target == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn_queries // max_num_target
        else:
            num_groups = self.num_groups
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def generate_dn_label_query(self, gt_labels: Tensor,
                                num_groups: int) -> Tensor:

        assert self.label_noise_scale > 0
        gt_labels_expand = gt_labels.repeat(2 * num_groups,
                                            1).view(-1)  # Note `* 2`  # noqa
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(
            -1)  # Note `* 0.5`
        new_labels = torch.randint_like(chosen_indice, 0, self.num_classes)
        noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice,
                                                       new_labels)
        if self.use_text_as_label:
            dn_label_query = self.label_mlp(self.text_embedding.to(noisy_labels_expand.device).float())[
                noisy_labels_expand]
        else:
            dn_label_query = self.label_embedding(noisy_labels_expand)
        return dn_label_query

    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:

        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly.
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query

    def collate_dn_queries(self, input_label_query: Tensor,
                           input_bbox_query: Tensor, batch_idx: Tensor,
                           batch_size: int, num_groups: int) -> Tuple[Tensor]:

        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat([
            torch.arange(num_target, device=device)
            for num_target in num_target_list
        ])
        map_query_index = torch.cat([
            map_query_index + max_num_target * i for i in range(2 * num_groups)
        ]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device)
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device)

        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str], num_matching_queries=None) -> Tensor:

        num_denoising_queries = int(max_num_target * 2 * num_groups)
        if num_matching_queries is None:
            num_matching_queries = self.num_matching_queries
        num_queries_total = num_denoising_queries + num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(int(max_num_target * 2 * i),
                              int(max_num_target * 2 * (i + 1)))
            left_scope = slice(int(max_num_target * 2 * i))
            right_scope = slice(int(max_num_target * 2 * (i + 1)),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask
