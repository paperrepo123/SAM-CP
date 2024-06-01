#############################################################saminst_head
# Copyright (c) OpenMMLab. All rights reserved.
import os.path
import os
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
import torch
from mmengine.structures import InstanceData, PixelData
from torch import Tensor
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmdet.models.utils import multi_apply, preprocess_panoptic_gt
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead
from mmdet.registry import TASK_UTILS
from scipy.optimize import linear_sum_assignment
from mmdet.models.losses import accuracy
from mmcv.cnn import Linear
import torch.nn as nn
import math
import copy
from mmengine.model import bias_init_with_prob, constant_init
import mmcv
from mmdet.evaluation.functional import INSTANCE_OFFSET
from mmdet.structures.mask import mask2bbox
from mmdet.models.layers import inverse_sigmoid
from mmcv.ops import batched_nms
from mmdet.models.layers import multiclass_nms


@MODELS.register_module()
class SamCPHead(DeformableDETRHead):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """

    def __init__(self, *args,
                 instace_cfg=dict(
                     category_embeddings='pretrained/clip/clip_categories_embedding/coco_80_with_bg_vit-b.pth',
                     num_thing_classes=80,
                 ),
                 semantic_cfg=dict(
                     category_embeddings='pretrained/clip/clip_categories_embedding/coco_semantic_53_vit-b.pth',
                     semantic_offline_overlap='pretrained/sam_proposals/coco_train/panoptic_overlaps/semantic/',
                     num_stuff_classes=53,
                 ),
                 use_patch_cls=False,
                 use_patch_query=True,
                 use_regression=False,
                 shared_cls=False,
                 use_cls_branch=True,
                 use_text_cls=True,
                 use_o2o_cls=False,
                 loss_affinity=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     reduction='mean',
                     loss_weight=20.0),
                 loss_dice=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=2.0)
                 , **kwargs):
        super().__init__(*args, **kwargs)
        self.use_patch_cls = use_patch_cls
        self.use_regression = use_regression
        self.loss_affinity = MODELS.build(loss_affinity)
        self.loss_dice = MODELS.build(loss_dice)
        self.use_cls_branch=use_cls_branch

        self.text_embedding = torch.load(instace_cfg.category_embeddings, map_location=torch.device('cpu'))

        self.num_classes = self.num_things_classes = instace_cfg.num_thing_classes
        self.semantic_cfg = semantic_cfg
        self.use_patch_query = use_patch_query
        self.use_o2o_cls = use_o2o_cls
        if self.semantic_cfg is not None:
            self.num_stuff_classes = semantic_cfg.num_stuff_classes
            self.num_classes = self.num_things_classes + self.num_stuff_classes
            self.semantic_offline_overlap = semantic_cfg.semantic_offline_overlap
            self.semantic_text_embedding = torch.load(semantic_cfg.category_embeddings,
                                                      map_location=torch.device('cpu'))
            self.text_embedding = torch.cat([self.text_embedding, self.semantic_text_embedding], dim=0)
            if self.loss_cls.use_sigmoid:
                self.cls_out_channels = self.num_classes
            else:
                self.cls_out_channels = self.num_classes + 1
        else:
            self.num_things_classes = self.num_classes
        cls_dim = self.text_embedding.shape[-1]

        self.use_text_cls = use_text_cls
        if self.use_text_cls:
            fc_cls = Linear(self.embed_dims, self.embed_dims)
        else:
            fc_cls = Linear(self.embed_dims, self.num_classes)
        if shared_cls:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])

        self.text_porj = nn.Linear(cls_dim, self.embed_dims, bias=True)
        self.logit_scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(cls_dim), requires_grad=True)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
        if self.train_cfg:
            assert 'assigner' in self.train_cfg, 'assigner should be provided ' \
                                                 'when train_cfg is set.'
            assigner = self.train_cfg['assigner']
            refine_assigner = self.train_cfg['refine_assigner']
            self.assigner_overlap = TASK_UTILS.build(assigner)
            self.assigner = TASK_UTILS.build(refine_assigner)
            if self.train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            for p in self.cls_branches.parameters():  ##init the weight
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)

        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        for m in self.reg_branches:
            nn.init.constant_(m[-1].bias.data[2:], 0.0)
        self.aa = 0
        self.pred_gt = 0
        self.pred_upper = 0
        self.gt_upper = 0
        self.cls_matched = 0
        self.bb = 0

    def loss(self, hidden_states: Tensor, references: Tensor, affinities: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor, patch_fts: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:

        batch_gt_instances = []
        batch_img_metas = []
        batch_proposal_patches = []
        batch_check_parts = []
        batch_gt_semantic_segs = []
        batch_check_parts_semantic = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_proposal_patches.append(data_sample.proposals)
            assign_result = self.assigner_overlap.assign(
                pred_instances=data_sample.proposals,
                gt_instances=data_sample.gt_instances,
                img_meta=data_sample.metainfo)
            batch_check_parts.append(assign_result.gt_inds)

            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)
        if self.semantic_cfg is not None:
            batch_gt_semantic = self.preprocess_gt(batch_gt_instances,
                                                   batch_gt_semantic_segs)
        else:
            batch_gt_semantic = None
        for i, data_sample in enumerate(batch_data_samples):
            if self.semantic_cfg is not None:
                file_name = self.semantic_offline_overlap + data_sample.img_path.split('/')[-1].split('.')[0] + '.pkl'
                if os.path.isfile(file_name):
                    semantic_offline_overlaps = torch.load(file_name, map_location=batch_check_parts[0].device)
                    semantic_overlap = semantic_offline_overlaps['segm']['iof'][:,
                                       data_sample.proposals[
                                           'proposals_valid_ids']] if 'proposals_valid_ids' in data_sample.proposals else \
                        semantic_offline_overlaps['segm']['iof']
                    box_overlap = semantic_overlap.new_zeros((self.num_classes, semantic_overlap.shape[-1]))
                    box_overlap[batch_gt_semantic[i]['labels'].long()] = bbox_overlaps(
                        data_sample.proposals['bboxes'].tensor, mask2bbox(batch_gt_semantic[i]['masks']),
                        mode='iof').permute(1, 0)

                    box_overlap = box_overlap[semantic_offline_overlaps['labels'].long()]
                    semantic_overlap = torch.stack([semantic_overlap, box_overlap]).min(0)[0]
                    # semantic_overlap = box_overlap
                    check_pt = semantic_overlap > self.assigner_overlap.pos_iou_thr
                    ov = semantic_offline_overlaps['segm']['iou'][:,
                         data_sample.proposals[
                             'proposals_valid_ids']] if 'proposals_valid_ids' in data_sample.proposals else \
                        semantic_offline_overlaps['segm']['iou']
                    check_pt[(ov > (ov * check_pt).max(1)[0][:, None]) * (ov > 0.1)] = 1
                    batch_check_parts_semantic.append(dict(overlap=semantic_overlap > self.assigner_overlap.pos_iou_thr,
                                                           labels=semantic_offline_overlaps['labels']))
                else:
                    batch_check_parts_semantic.append(
                        dict(overlap=batch_check_parts[i].new_zeros((0, batch_check_parts[i].shape[0])),
                             labels=batch_check_parts[i].new_zeros((0,))))
            else:
                batch_check_parts_semantic.append(None)
        all_layers_outputs_classes, references, patch_outputs_classes = self(hidden_states,
                                                                             patch_fts if self.use_patch_cls else None,
                                                                             references)

        loss_inputs = (
            all_layers_outputs_classes, references, affinities, patch_outputs_classes, enc_outputs_class,
            enc_outputs_coord, batch_gt_instances, batch_gt_semantic, batch_img_metas, batch_proposal_patches,
            batch_check_parts, batch_check_parts_semantic, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def preprocess_gt(
            self, batch_gt_instances: InstanceList,
            batch_gt_semantic_segs: List[Optional[PixelData]]) -> InstanceList:

        num_things_list = [self.num_things_classes] * len(batch_gt_instances)
        num_stuff_list = [self.num_stuff_classes] * len(batch_gt_instances)
        gt_labels_list = [
            gt_instances['labels'] for gt_instances in batch_gt_instances
        ]
        gt_masks_list = [
            gt_instances['masks'] for gt_instances in batch_gt_instances
        ]
        gt_semantic_segs = [
            None if gt_semantic_seg is None else gt_semantic_seg.sem_seg
            for gt_semantic_seg in batch_gt_semantic_segs
        ]

        targets = multi_apply(self.preprocess_semantic_gt, gt_semantic_segs, num_things_list,
                              num_stuff_list)
        masks, labels = targets
        batch_gt_instances = [
            InstanceData(labels=label, masks=mask)
            for label, mask in zip(labels, masks)
        ]
        return batch_gt_instances

    def preprocess_semantic_gt(self, gt_semantic_seg, num_things,
                               num_stuff):
        num_classes = num_things + num_stuff
        gt_semantic_seg = gt_semantic_seg.squeeze(0)

        semantic_labels = torch.unique(
            gt_semantic_seg,
            sorted=False,
            return_inverse=False,
            return_counts=False)
        stuff_masks_list = []
        stuff_labels_list = []
        for label in semantic_labels:
            if label < num_things or label >= num_classes:
                continue
            stuff_mask = gt_semantic_seg == label
            stuff_masks_list.append(stuff_mask)
            stuff_labels_list.append(label)

        if len(stuff_masks_list) > 0:
            stuff_masks = torch.stack(stuff_masks_list, dim=0)
            stuff_labels = torch.stack(stuff_labels_list, dim=0)
            stuff_masks = stuff_masks.long()
        else:
            stuff_labels = gt_semantic_seg.new_zeros((0,))
            stuff_masks = gt_semantic_seg.new_zeros((0, *gt_semantic_seg.shape[-2:]))

        return stuff_masks, stuff_labels

    def forward(self, hidden_states: Tensor, patch_fts: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        text_features = self.text_embedding.to(hidden_states[0].device).float()[:self.num_classes]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_feat_bias = torch.matmul(text_features, self.bias_lang) + self.bias0
        text_features = self.text_porj(text_features / 0.1)  # (8, 256)

        for layer_id in range(hidden_states.shape[0]):
            # reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            if self.use_cls_branch:
                outputs_feat = self.cls_branches[layer_id](hidden_state)
            else:
                outputs_feat=hidden_state
            if self.use_text_cls:
                outputs_feat = outputs_feat / outputs_feat.norm(dim=-1, keepdim=True)
                bias = text_feat_bias.unsqueeze(0).repeat(outputs_feat.shape[0], outputs_feat.shape[1], 1)

                outputs_class = (outputs_feat @ text_features.t() / self.logit_scale.exp()) + bias
                MAX_CLAMP_VALUE = 50000
                outputs_class = torch.clamp(
                    outputs_class, max=MAX_CLAMP_VALUE)
                outputs_class = torch.clamp(
                    outputs_class, min=-MAX_CLAMP_VALUE)
            else:
                outputs_class = outputs_feat
            all_layers_outputs_classes.append(outputs_class)
            reference = inverse_sigmoid(references[layer_id])
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_coords.append(outputs_coord)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        if patch_fts is not None:
            patch_outputs_feat = self.cls_branches[-1](patch_fts)
            patch_outputs_feat = patch_outputs_feat / patch_outputs_feat.norm(dim=-1, keepdim=True)
            bias = text_feat_bias.unsqueeze(0).repeat(patch_outputs_feat.shape[0], patch_outputs_feat.shape[1], 1)
            patch_outputs_classes = (patch_outputs_feat @ text_features.t() / self.logit_scale.exp()) + bias
        else:
            patch_outputs_classes = None
        return all_layers_outputs_classes, all_layers_outputs_coords, patch_outputs_classes

    def loss_by_feat(
            self,
            all_layers_cls_scores: Tensor,
            all_layers_references: Tensor,
            all_layers_affinities: Tensor,
            patch_outputs_classes: Tensor,
            enc_cls_scores: Tensor,
            enc_bbox_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_gt_semantic: InstanceList,
            batch_img_metas: List[dict],
            batch_proposal_patches: List[dict],
            batch_check_parts: List[Tensor],
            batch_check_parts_semantic: List[Tensor],
            dn_meta: Dict[str, int],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:

        # extract denoising and matching part of outputs
        if dn_meta is not None:
            (all_layers_matching_cls_scores, all_layers_matching_references, all_layers_matching_affinities,
             all_layers_denoising_cls_scores, all_layers_denoising_references, all_layers_denoising_bbox_preds) = \
                self.split_outputs(
                    all_layers_cls_scores, all_layers_references, all_layers_affinities, dn_meta)
        else:
            all_layers_matching_cls_scores, all_layers_matching_references, all_layers_matching_affinities = \
                all_layers_cls_scores, all_layers_references, all_layers_affinities
            all_layers_denoising_cls_scores, all_layers_denoising_references, all_layers_denoising_bbox_preds = None, None, None
        # loss_dict = super().loss_by_feat().loss_by_feat(
        #     all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
        #     batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        if self.semantic_cfg is not None:
            all_layers_instance_cls_scores, all_layers_instance_references, all_layers_instance_affinities, \
                all_layers_semantic_cls_scores, all_layers_smantic_references, all_layers_semantic_affinities = \
                self.split_outputs(all_layers_matching_cls_scores, all_layers_matching_references,
                                   all_layers_matching_affinities,
                                   dict(num_denoising_queries=len(self.semantic_text_embedding)))
        else:
            all_layers_instance_cls_scores, all_layers_instance_references, all_layers_instance_affinities = \
                all_layers_matching_cls_scores, all_layers_matching_references, all_layers_matching_affinities

        loss_dict = dict()
        if self.use_regression:
            all_layers_instance_ref_cls_scores = all_layers_instance_cls_scores[-1]
            all_layers_instance_ref_references = all_layers_instance_references[-1]
            all_layers_instance_ref_affinities = all_layers_instance_affinities[-1]
            all_layers_instance_cls_scores = all_layers_instance_cls_scores[:-1]
            all_layers_instance_references = all_layers_instance_references[:-1]
            all_layers_instance_affinities = all_layers_instance_affinities[:-1]
            if dn_meta is not None:
                all_layers_denoising_cls_scores = all_layers_denoising_cls_scores[:-1]
                all_layers_denoising_references = all_layers_denoising_references[:-1]
                all_layers_denoising_bbox_preds = all_layers_denoising_bbox_preds[:-1]
            if self.semantic_cfg is not None:
                all_layers_semantic_cls_scores = all_layers_semantic_cls_scores[:-1]
                all_layers_smantic_references = all_layers_smantic_references[:-1]
                all_layers_semantic_affinities = all_layers_semantic_affinities[:-1]

        if self.semantic_cfg is not None:
            losses_semantic_cls, semantic_acc, losses_semantic_affinity, losses_semantic_dice = multi_apply(
                self.loss_by_feat_semantic_single,
                all_layers_semantic_cls_scores,
                all_layers_smantic_references,
                all_layers_semantic_affinities,
                batch_gt_semantic=batch_gt_semantic,
                batch_img_metas=batch_img_metas,
                batch_proposal_patches=batch_proposal_patches,
                batch_check_parts=batch_check_parts_semantic)
            loss_dict['loss_sem_dice'] = losses_semantic_dice[-1]
            loss_dict['loss_sem_affinity'] = losses_semantic_affinity[-1]
            loss_dict['loss_sem_cls'] = losses_semantic_cls[-1]
            loss_dict['sem_Acc'] = semantic_acc[-1]

            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, acc_i, loss_affinity_i, loss_dice_i in \
                    zip(losses_semantic_cls[:-1], semantic_acc[:-1], losses_semantic_affinity[:-1],
                        losses_semantic_dice[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_sem_dice'] = loss_dice_i
                loss_dict[f'd{num_dec_layer}.loss_sem_affinity'] = loss_affinity_i
                loss_dict[f'd{num_dec_layer}.loss_sem_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.sem_Acc'] = acc_i

                num_dec_layer += 1
        if self.use_patch_cls:
            losses_cls_patch, acc_patch = self.loss_by_feat_patch(patch_outputs_classes,
                                                                  batch_gt_instances=batch_gt_instances,
                                                                  batch_img_metas=batch_img_metas,
                                                                  batch_proposal_patches=batch_proposal_patches,
                                                                  batch_check_parts=batch_check_parts)

            loss_dict['loss_cls_patch'] = losses_cls_patch
            loss_dict['Acc_patch'] = acc_patch

        if self.use_regression:
            use_regression = [False for aa in all_layers_instance_affinities]
            # use_regression[-1] = True
        else:
            use_regression = [False for aa in all_layers_instance_affinities]
        losses_cls, acc, losses_affinity, loss_dice, loss_iou, loss_bbox = multi_apply(
            self.loss_by_feat_instance_single,
            all_layers_instance_cls_scores,
            all_layers_instance_references,
            all_layers_instance_affinities,
            use_regression,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_proposal_patches=batch_proposal_patches,
            batch_check_parts=batch_check_parts)

        # loss from the last decoder layer
        loss_dict['loss_dice'] = loss_dice[-1]
        loss_dict['loss_affinity'] = losses_affinity[-1]
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['Acc'] = acc[-1]

        if self.use_regression:
            losses_cls_refine, losses_bbox_refine, losses_iou_refine, acc_refine = multi_apply(
                self.loss_by_feat_single,
                [all_layers_instance_ref_cls_scores],
                [all_layers_instance_ref_references],
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas)
            loss_dict['loss_cls_refine'] = losses_cls_refine
            loss_dict['loss_iou_refine'] = losses_iou_refine
            loss_dict['loss_bbox_refine'] = losses_bbox_refine
            loss_dict['Acc'] = acc_refine

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, acc_i, loss_affinity_i, loss_dice_i in \
                zip(losses_cls[:-1], acc[:-1], losses_affinity[:-1], loss_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_affinity'] = loss_affinity_i
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.Acc'] = acc_i

            num_dec_layer += 1

        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_acc, dn_losses_affinity, dn_loss_dice = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                batch_proposal_patches=batch_proposal_patches,
                batch_check_parts=batch_check_parts,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_dice'] = dn_loss_dice[-1]
            loss_dict['dn_loss_affinity'] = dn_losses_affinity[-1]
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_acc'] = dn_acc[-1]
            for num_dec_layer, (dn_loss_cls_i, dn_loss_affinity_i, dn_loss_dice_i, dn_acc_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_affinity[:-1],
                                  dn_loss_dice[:-1], dn_acc[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_dice'] = dn_loss_dice_i
                loss_dict[f'd{num_dec_layer}.dn_loss_affinity'] = dn_loss_affinity_i
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = dn_loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_acc'] = dn_acc_i

        return loss_dict

    def loss_by_feat_semantic_single(self, cls_scores: Tensor, references: Tensor, affinity_preds: Tensor,
                                     batch_gt_semantic: InstanceList,
                                     batch_img_metas: List[dict],
                                     batch_proposal_patches: List[dict],
                                     batch_check_parts: List[Tensor]):
        cls_reg_targets = self.get_semantic_targets(cls_scores, affinity_preds, batch_proposal_patches,
                                                    batch_check_parts,
                                                    batch_gt_semantic, batch_img_metas)
        (labels_list, label_weights_list, affinity_targets_list, affinity_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        affinity_targets = torch.cat(affinity_targets_list, 0)
        affinity_weights = torch.cat(affinity_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            # cls_scores[torch.arange(self.num_stuff_classes), self.num_things_classes + torch.arange(
            #     self.num_stuff_classes)].shape
            if self.use_o2o_cls:
                loss_cls = self.loss_cls(
                    cls_scores[torch.arange(len(cls_scores)), self.num_things_classes + torch.arange(
                        self.num_stuff_classes).repeat(len(labels_list)), None], (labels == self.num_classes).long(),
                    label_weights, avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        acc = accuracy(cls_scores[pos_inds], labels[pos_inds])

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        ## loss affinity

        affinity_preds = affinity_preds.reshape(-1, affinity_preds.shape[-1])
        loss_affinity = self.loss_affinity(affinity_preds, affinity_targets, affinity_weights,
                                           avg_factor=num_total_pos)
        affinity_preds_pos = affinity_preds[pos_inds]
        affinity_preds_pos[affinity_weights[pos_inds] == 0] = inverse_sigmoid(affinity_targets[pos_inds][
                                                                                  affinity_weights[
                                                                                      pos_inds] == 0].float())
        loss_dice = self.loss_dice(affinity_preds_pos, affinity_targets[pos_inds],
                                   avg_factor=num_total_pos)

        return loss_cls, acc, loss_affinity, loss_dice

    def get_semantic_targets(self, dn_cls_scores, dn_affinity_preds, batch_proposal_patches, batch_check_parts,
                             batch_gt_semantic: InstanceList,
                             batch_img_metas: dict) -> tuple:

        (labels_list, label_weights_list, affinity_targets_list, affinity_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_semantic_targets_single,
            dn_cls_scores,
            dn_affinity_preds,
            batch_proposal_patches,
            batch_check_parts,
            batch_gt_semantic,
            batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, affinity_targets_list,
                affinity_weights_list, num_total_pos, num_total_neg)

    def _get_semantic_targets_single(self, cls_scores, affinity_preds, batch_proposal_patches,
                                     check_parts_dict, gt_semantic: InstanceData,
                                     img_meta: dict) -> tuple:

        gt_labels = gt_semantic.labels
        # check_parts = check_parts.permute(1, 0)
        num_semantic = self.num_stuff_classes
        device = cls_scores.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            pos_inds = gt_labels.long() - self.num_things_classes
            pos_assigned_gt_inds = t.flatten()

        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_labels.new_tensor([], dtype=torch.long)

        check_parts = check_parts_dict['overlap'].long()
        check_parts_labels = check_parts_dict['labels']
        valid = check_parts >= 0
        check_parts = check_parts.masked_fill(~valid, 0)

        flg = torch.ones(self.num_stuff_classes)
        flg[pos_inds] = 0
        neg_inds = flg.nonzero()

        # label targets
        labels = gt_labels.new_full((num_semantic,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds].long()
        label_weights = gt_labels.new_ones(num_semantic).float()

        # affinity targets
        num_patches = batch_proposal_patches.bboxes.shape[0]
        affinity_targets = torch.zeros_like(affinity_preds, dtype=torch.long, device=device)
        affinity_weights = torch.zeros_like(affinity_targets, dtype=torch.float, device=device)
        affinity_weights[pos_inds, :num_patches] = 1.0

        affinity_targets[(check_parts_labels - self.num_things_classes).long(), :num_patches] = check_parts.long()
        affinity_targets = (affinity_targets * affinity_weights).long()

        affinity_weights[(check_parts_labels - self.num_things_classes).long(), :num_patches] *= valid
        label_weights[(check_parts_labels - self.num_things_classes).long()] *= check_parts.max(1)[0][:, None].float()[
                                                                                :, 0]
        # affinity_weights[(check_parts_labels - self.num_things_classes).long()] *= check_parts.max(1)[0][:,
        #                                                                            None].float()
        labels[(check_parts_labels - self.num_things_classes).long()[
            (1 - check_parts.max(1)[0][:, None][:, 0]).bool()]] = self.num_classes
        if self.train_cfg.get('query_assigner_cfg', None).get('weighted', False):
            wt = affinity_targets.sum(-1, keepdim=True)
            wt = wt.masked_fill(wt == 0, 1)
            affinity_weights = affinity_weights / wt
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        pos_inds = pos_inds.nonzero()
        # pos_inds*=check_parts.max(1)[0][:, None].float()[  :, 0]
        return (labels, label_weights, affinity_targets, affinity_weights, pos_inds,
                neg_inds)

    def loss_by_feat_patch(self, patch_outputs_classes, batch_gt_instances: InstanceList,
                           batch_img_metas: List[dict],
                           batch_proposal_patches: List[dict], batch_check_parts):
        labels_list = []
        label_weights_list = []
        for i in range(len(batch_proposal_patches)):
            patch_cls = patch_outputs_classes[i]
            proposal_patches = batch_proposal_patches[i]
            gt_instances = batch_gt_instances[i]
            img_metas = batch_img_metas[i]

            num_patches = len(batch_proposal_patches[i].bboxes)

            labels = patch_cls.new_full((len(patch_cls),),
                                        self.num_classes,
                                        dtype=torch.long)
            if gt_instances.labels.shape[0] > 0:
                labels[:num_patches][batch_check_parts[i].max(-1)[0] > 0] = \
                    gt_instances.labels[batch_check_parts[i].argmax(1)][batch_check_parts[i].max(-1)[0] > 0]
            label_weights = patch_cls.new_ones(len(patch_cls))
            labels_list.append(labels)
            label_weights_list.append(label_weights)
        labels_list = torch.cat(labels_list)
        label_weights_list = torch.cat(label_weights_list)
        patch_outputs_classes = patch_outputs_classes.reshape(-1, self.cls_out_channels)
        pos_inds = (labels_list >= 0) & (labels_list < self.num_classes)
        cls_avg_factor = pos_inds.sum()
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                patch_outputs_classes.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(patch_outputs_classes, labels_list, label_weights_list, avg_factor=cls_avg_factor)
        acc = accuracy(patch_outputs_classes[pos_inds], labels_list[pos_inds])
        return loss_cls, acc

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_affinity_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                batch_proposal_patches: List[dict],
                batch_check_parts: List[Tensor],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:

        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_affinity_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_proposal_patches=batch_proposal_patches,
            batch_check_parts=batch_check_parts,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_affinity_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        batch_proposal_patches: List[dict],
                        batch_check_parts: List[Tensor],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:

        cls_reg_targets = self.get_dn_targets(dn_cls_scores, dn_affinity_preds, batch_proposal_patches,
                                              batch_check_parts,
                                              batch_gt_instances, batch_img_metas, dn_meta)
        (labels_list, label_weights_list, affinity_targets_list, affinity_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        affinity_targets = torch.cat(affinity_targets_list, 0)
        affinity_weights = torch.cat(affinity_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)
        pos_inds = (labels >= 0) & (labels < self.num_classes)
        acc = accuracy(cls_scores[pos_inds], labels[pos_inds])

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        ## loss affinity

        dn_affinity_preds = dn_affinity_preds.reshape(-1, dn_affinity_preds.shape[-1])
        loss_affinity = self.loss_affinity(dn_affinity_preds, affinity_targets, affinity_weights,
                                           avg_factor=num_total_pos)
        # dn_affinity_preds[pos_inds][affinity_weights[pos_inds] > 0] = affinity_targets[pos_inds][
        #     affinity_weights[pos_inds] > 0].float()
        loss_dice = self.loss_dice(dn_affinity_preds, affinity_targets, pos_inds,
                                   avg_factor=num_total_pos)

        return loss_cls, acc, loss_affinity, loss_dice

    def get_dn_targets(self, dn_cls_scores, dn_affinity_preds, batch_proposal_patches, batch_check_parts,
                       batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
            int]) -> tuple:
        (labels_list, label_weights_list, affinity_targets_list, affinity_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_dn_targets_single,
            dn_cls_scores,
            dn_affinity_preds,
            batch_proposal_patches,
            batch_check_parts,
            batch_gt_instances,
            batch_img_metas,
            dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, affinity_targets_list,
                affinity_weights_list, num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, dn_cls_scores,
                               dn_affinity_preds, batch_proposal_patches,
                               check_parts, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
            int]) -> tuple:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        check_parts = check_parts.permute(1, 0)
        valid = check_parts >= 0
        check_parts = check_parts.masked_fill(~valid, 0)

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # affinity targets
        num_patches = batch_proposal_patches.bboxes.shape[0]
        affinity_targets = torch.zeros_like(dn_affinity_preds, dtype=torch.long, device=device)
        affinity_weights = torch.zeros_like(dn_affinity_preds, device=device)
        affinity_weights[pos_inds, :num_patches] = 1.0

        affinity_targets[pos_inds, :num_patches] = check_parts.repeat([num_groups, 1])

        affinity_weights[pos_inds, :num_patches] *= valid.repeat([num_groups, 1])
        label_weights[pos_inds] = check_parts.max(1)[0][:, None].float().repeat([num_groups, 1])[:, 0]
        affinity_weights[pos_inds] *= check_parts.max(1)[0][:, None].float().repeat([num_groups, 1])
        if self.train_cfg.get('query_assigner_cfg', None).get('weighted', False):
            affinity_weights = affinity_weights / (affinity_targets.sum(-1, keepdim=True) + 1e-10)
        pos_inds = pos_inds[check_parts.max(1)[0][:, None].float().repeat([num_groups, 1])[:, 0].bool()]

        return (labels, label_weights, affinity_targets, affinity_weights, pos_inds,
                neg_inds)

    def loss_by_feat_instance_single(self, cls_scores: Tensor, references: Tensor, affinity_preds: Tensor,
                                     use_regression,
                                     batch_gt_instances: InstanceList,
                                     batch_img_metas: List[dict],
                                     batch_proposal_patches: List[dict],
                                     batch_check_parts: List[Tensor]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i][:, :self.num_things_classes] for i in range(num_imgs)]
        references_list = [references[i] for i in range(num_imgs)]
        affinity_list = [affinity_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_instance_targets(cls_scores_list, affinity_list,
                                                    batch_gt_instances, batch_img_metas, batch_proposal_patches,
                                                    batch_check_parts)
        (labels_list, label_weights_list, affiniy_targets_list, bbox_targets_list, affiniy_weights_list,
         num_total_pos, num_total_neg, ignore_flag) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        affiniy_targets = torch.cat(affiniy_targets_list, 0)
        affiniy_weights = torch.cat(affiniy_weights_list, 0)

        ignore_flag = torch.cat(ignore_flag, 0)
        pos_inds = (labels >= 0) & (labels < self.num_things_classes)
        if use_regression:
            bbox_weights = (affiniy_weights[:, 0, None] > 0).float().repeat(1, 4)
            label_weights = ignore_flag
            affiniy_weights *= ignore_flag[:, None]
            pos_inds = pos_inds
        else:
            label_weights *= ignore_flag
            affiniy_weights *= ignore_flag[:, None]
            pos_inds *= ignore_flag.bool()
        num_total_pos_cls = pos_inds.sum()

        # classification loss
        cls_scores = cls_scores[..., :self.num_things_classes].reshape(-1, self.num_things_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos_cls * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        acc = accuracy(cls_scores[pos_inds], labels[pos_inds])

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        affinity_preds = affinity_preds.reshape(-1, affinity_preds.shape[-1])
        loss_afinity = self.loss_affinity(affinity_preds, affiniy_targets, affiniy_weights * ignore_flag[:, None],
                                          avg_factor=num_total_pos)
        # affinity_preds[pos_inds][affiniy_weights[pos_inds] > 0] = affiniy_targets[pos_inds][
        #     affiniy_weights[pos_inds] > 0].float()

        loss_dice = self.loss_dice(affinity_preds, affiniy_targets, affiniy_targets.max(-1)[0] > 0,
                                   avg_factor=num_total_pos)
        # from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
        # pos_inds = (labels >= 0) & (labels < self.num_classes)
        # cls_matched = cls_scores[pos_inds].sigmoid()[torch.arange(pos_inds.sum()), labels[pos_inds]]
        # pred_gt = bbox_overlaps(bbox_merged[matched_row_inds], gt_instances.bboxes[matched_col_inds], is_aligned=True)
        # pred_upper = bbox_overlaps(bbox_merged[matched_row_inds],
        #                            self.get_bboxes_segms(check_parts, batch_proposal_patches, cls_score,
        #                                                  img_shape,
        #                                                  self.train_cfg.get('with_score_norm', False),
        #                                                  thre=0.5)[0][matched_col_inds], is_aligned=True)
        # gt_upper = ck[matched_col_inds]
        # cls_matched = cls_score.sigmoid()[:, :self.num_things_classes][matched_row_inds][
        #     torch.arange(len(matched_row_inds)), gt_instances.labels[matched_col_inds]]
        # idx = gt_upper > 0.5
        # self.pred_gt += pred_gt[idx].sum()
        # self.pred_upper += pred_upper[idx].sum()
        # self.gt_upper += gt_upper[idx].sum()
        # self.cls_matched += cls_matched[idx].sum()
        # self.aa += len(matched_row_inds[idx])
        # self.bb += len(matched_row_inds)
        #
        # print('pred_gt', self.pred_gt / self.aa, 'pred_upper', self.pred_upper / self.aa,
        #       'gt_upper', self.gt_upper / self.aa, 'cls_matched', self.cls_matched / self.aa,
        #       'missing', self.aa / self.bb)

        if use_regression:
            factors = []
            for img_meta, bbox_pred in zip(batch_img_metas, references):
                img_h, img_w, = img_meta['img_shape']
                factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                               img_h]).unsqueeze(0).repeat(
                    bbox_pred.size(0), 1)
                factors.append(factor)
            factors = torch.cat(factors, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_preds = torch.cat(references_list, 0)

            bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
            bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

            # regression IoU loss, defaultly GIoU loss
            loss_iou = self.loss_iou(
                bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

            # regression L1 loss
            loss_bbox = self.loss_bbox(
                bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
            # from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
            # # pos_inds = (labels >= 0) & (labels < self.num_classes)
            # cls_matched = cls_scores[pos_inds].sigmoid()[torch.arange(pos_inds.sum()), labels[pos_inds]]
            # pred_gt = bbox_overlaps(bboxes[pos_inds], bboxes_gt[pos_inds], is_aligned=True)
            #
            # self.pred_gt += pred_gt.sum()
            # self.cls_matched += cls_matched.sum()
            # self.aa += pos_inds.sum()
            #
            # print('pred_gt', self.pred_gt / self.aa, 'cls_matched', self.cls_matched / self.aa)
        else:
            loss_iou, loss_bbox = None, None

        return loss_cls, acc, loss_afinity, loss_dice, loss_bbox, loss_iou

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_references: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_denoising_references = \
                all_layers_references[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
            all_layers_matching_references = \
                all_layers_references[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_references = None
            all_layers_denoising_bbox_preds = None

            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_references = all_layers_references
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_references, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores, all_layers_denoising_references, all_layers_denoising_bbox_preds)

    def get_instance_targets(self, cls_scores_list: List[Tensor],
                             affinity_preds_list: List[Tensor],
                             batch_gt_instances: InstanceList,
                             batch_img_metas: List[dict],
                             batch_proposal_patches: List[dict],
                             batch_check_parts: List[Tensor]) -> tuple:
        (labels_list, label_weights_list, affinity_targets_list, bbox_targets_list, affinities_weights_list,
         pos_inds_list, neg_inds_list, ignore_flag) = multi_apply(self._get_instance_targets_single,
                                                                  cls_scores_list, affinity_preds_list,
                                                                  batch_gt_instances, batch_img_metas,
                                                                  batch_proposal_patches, batch_check_parts)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, affinity_targets_list, bbox_targets_list,
                affinities_weights_list, num_total_pos, num_total_neg, ignore_flag)

    def _get_instance_targets_single(self, cls_score: Tensor, affinity_pred: Tensor,
                                     gt_instances: InstanceData,
                                     img_meta: dict,
                                     batch_proposal_patches: InstanceData,
                                     check_parts: Tensor) -> tuple:
        img_h, img_w = img_meta['img_shape']
        num_patches = batch_proposal_patches.bboxes.shape[0]
        num_quries = len(cls_score)

        img_shape = img_meta['img_shape']
        factor = affinity_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)

        check_parts = check_parts.permute(1, 0)
        valid = check_parts >= 0
        check_parts = check_parts.masked_fill(~valid, 0)
        query_ass_cfg = self.train_cfg.get('query_assigner_cfg', None)
        if query_ass_cfg is not None:
            ass_type = query_ass_cfg['ass_type']
            is_weighted = query_ass_cfg.get('weighted', False)
            if ass_type == 'one2one':
                ass_cost = query_ass_cfg['ass_cost']
                affinity_cost_weight, cls_cost_weight, bbox_cost_weight, iou_cost_weight, dice_cost_weight = \
                    ass_cost.get('affinity_cost', 50.0), ass_cost.get('cls_cost', 2.0), \
                        ass_cost.get('bbox_cost', 5.0), ass_cost.get('iou_cost', 2.0), ass_cost.get('dice_cost', 2.0)
        else:
            ass_type = 'one2one'
            affinity_cost_weight, cls_cost_weight, bbox_cost_weight, iou_cost_weight = 50.0, 2.0, 5.0, 2.0
            is_weighted = False

        if ass_type == 'fix_ass':
            img_shape = img_meta['img_shape']
            bbox_merged, _, affinity_pred_sigmoid, valid_idx = self.get_bboxes_segms(
                affinity_pred[:num_patches, :num_patches],
                batch_proposal_patches,
                cls_score,
                img_shape, self.train_cfg.get(
                    'with_score_norm', False),
                thre=0.5,
                return_valid=True,
                is_patch_query=False)
            matched_row_inds, matched_col_inds = [], []
            matched_row_inds_cls, matched_col_inds_cls = [], []
            for jj, cp in enumerate(check_parts):
                if cp.sum() > 0:
                    part_bbox = bbox_merged[cp.nonzero()[:, 0]]
                    iou = bbox_overlaps(part_bbox, gt_instances.bboxes[jj, None])
                    row_idx = cp.nonzero()[:, 0][(bbox_overlaps(part_bbox, part_bbox[iou.max(0)[1]]) > 0.8)[:, 0]]
                    col_idx = cp.new_full((len(row_idx),), jj, dtype=torch.long, device=cp.device)
                    matched_row_inds_cls.extend(row_idx)
                    matched_col_inds_cls.extend(col_idx)
                    matched_row_inds.extend(cp.nonzero()[:, 0])
                    matched_col_inds.extend(
                        cp.new_full((len(cp.nonzero()[:, 0]),), jj, dtype=torch.long, device=cp.device))
            if len(matched_col_inds) > 0:
                matched_col_inds = torch.stack(matched_col_inds)
                matched_row_inds = torch.stack(matched_row_inds)
                matched_col_inds_cls = torch.stack(matched_col_inds_cls)
                matched_row_inds_cls = torch.stack(matched_row_inds_cls)
            else:
                matched_col_inds = affinity_pred.new_zeros((0,), dtype=torch.long)
                matched_row_inds = affinity_pred.new_zeros((0,), dtype=torch.long)
                matched_col_inds_cls = affinity_pred.new_zeros((0,), dtype=torch.long)
                matched_row_inds_cls = affinity_pred.new_zeros((0,), dtype=torch.long)

        if ass_type == 'one2one':
            img_shape = img_meta['img_shape']
            bbox_merged, _, affinity_pred_sigmoid, valid_idx = self.get_bboxes_segms(
                affinity_pred[:num_patches, :num_patches],
                batch_proposal_patches,
                cls_score,
                img_shape, self.train_cfg.get(
                    'with_score_norm', False),
                thre=0.5,
                return_valid=True,
                is_patch_query=False)
            cal_cls_cost = TASK_UTILS.build(dict(type='FocalLossCost', weight=cls_cost_weight))
            # bbox_cost = TASK_UTILS.build(dict(type='BBoxL1Cost', weight=bboxcost_weight))
            cal_affi_cost = TASK_UTILS.build(dict(type='FocalLossCost', weight=affinity_cost_weight, binary_input=True))
            cal_iou_cost = TASK_UTILS.build(dict(type='IoUCost', iou_mode='giou', weight=iou_cost_weight))
            cal_dice_cost = TASK_UTILS.build(dict(type='DiceCost', weight=dice_cost_weight, pred_act=True, eps=1.0))
            cls_cost = cal_cls_cost(InstanceData(scores=cls_score[:num_patches]), gt_instances)
            affinity_to_cal = affinity_pred_sigmoid[:num_patches, :num_patches] if self.train_cfg.get(
                'with_score_norm_aff_dice', False) else affinity_pred[:num_patches, :num_patches]
            affinity_cost = cal_affi_cost(InstanceData(masks=affinity_to_cal), InstanceData(masks=check_parts))
            dice_cost = cal_dice_cost(InstanceData(masks=affinity_to_cal), InstanceData(masks=check_parts))
            ## before 0920 11.20 , ioucost is 2.0

            if is_weighted:
                flag = check_parts.sum(1)
                flag = flag.masked_fill(flag == 0, 1)
                affinity_cost = affinity_cost * num_patches / (flag)

            img_h, img_w = img_shape
            factor = gt_instances.bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            bbox_cost = torch.cdist(bbox_merged / factor, gt_instances.bboxes / factor, p=1) * bbox_cost_weight
            iou_cost = cal_iou_cost(InstanceData(bboxes=bbox_merged), gt_instances)
            match_cost = cls_cost + affinity_cost + dice_cost + bbox_cost + iou_cost
            # match_cost = match_cost / (check_parts.max(0)[0] + 1e-8)
            match_cost = match_cost.detach().cpu()
            if self.use_patch_query:
                match_cost = match_cost[:num_patches]

            matched_row_inds, matched_col_inds = linear_sum_assignment(match_cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                cls_score.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                cls_score.device)
            if self.train_cfg['query_assigner_cfg'].get('add_one2many', False):
                from mmdet.models.task_modules.assigners import MaxIoUAssigner
                iou_ass = MaxIoUAssigner(
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.8,
                    min_pos_iou=0.8,
                    match_low_quality=False,
                    ignore_iof_thr=-1)
                ass_res = iou_ass.assign(InstanceData(priors=bbox_merged),
                                         InstanceData(bboxes=bbox_merged[matched_row_inds],
                                                      labels=gt_instances.labels[matched_col_inds]))
                matched_row_inds_ = (ass_res.gt_inds > 0).nonzero()[:, 0]
                matched_col_inds_ = ass_res.gt_inds[matched_row_inds_] - 1
                # matched_row_inds = torch.cat([matched_row_inds, matched_row_inds_])
                # matched_col_inds = torch.cat([matched_col_inds, matched_col_inds_])



        elif ass_type == 'one2many':
            pass
            img_shape = img_meta['img_shape']
            bbox_merged = self.get_bboxes_segms(affinity_pred[:, :num_patches], batch_proposal_patches, cls_score,
                                                img_shape, self.train_cfg.get('with_score_norm', False), thre=0.0)[0]
            bbox_iou = bbox_overlaps(bbox_merged, gt_instances.bboxes)
            #########################
            from mmdet.models.task_modules.assigners import MaxIoUAssigner
            iou_ass = MaxIoUAssigner(
                pos_iou_thr=0.8,
                neg_iou_thr=0.8,
                min_pos_iou=0.8,
                match_low_quality=False,
                ignore_iof_thr=-1)
            ass_res = iou_ass.assign(InstanceData(priors=bbox_merged), gt_instances)
            matched_row_inds = (ass_res.gt_inds > 0).nonzero()[:, 0]
            matched_col_inds = ass_res.gt_inds[matched_row_inds] - 1
            #########################

        ck = bbox_overlaps(gt_instances.bboxes, self.get_bboxes_segms(check_parts, batch_proposal_patches, cls_score,
                                                                      img_shape,
                                                                      self.train_cfg.get('with_score_norm', False),
                                                                      thre=0.5)[0], is_aligned=True)

        labels = affinity_pred.new_full((num_quries,),
                                        self.num_things_classes,
                                        dtype=torch.long)
        if ass_type == 'fix_ass':
            labels[matched_row_inds_cls] = gt_instances.labels[matched_col_inds_cls]
            label_weights = affinity_pred.new_ones(num_quries)
            # label_weights[matched_row_inds_] = 0
            label_weights[num_patches:] = 0
        else:
            labels[matched_row_inds] = gt_instances.labels[matched_col_inds]
            label_weights = affinity_pred.new_ones(num_quries)
            # label_weights[matched_row_inds_] = 0
            label_weights[num_patches:] = 0

        affinity_targets = affinity_pred.new_zeros((num_quries, num_quries)).long()
        affinity_weights = affinity_pred.new_zeros((num_quries, num_quries))
        affinity_weights[matched_row_inds, :num_patches] = 1.0

        affinity_targets[matched_row_inds, :num_patches] = check_parts[matched_col_inds]

        affinity_weights[matched_row_inds, :num_patches] *= valid[matched_col_inds]

        neg_inds = (labels == self.num_things_classes).nonzero()[:, 0]
        ignore_flag = affinity_pred.new_ones(num_quries)
        ignore_flag[matched_row_inds] = (check_parts.max(1)[0].float() * (ck > 0.5))[matched_col_inds]
        pos_inds = matched_row_inds

        # label_weights[matched_row_inds] = (check_parts.max(1)[0].float() * (ck > 0.5))[matched_col_inds]
        # affinity_weights[matched_row_inds] *= ((check_parts.max(1)[0] * (ck > 0.5))[:, None].float())[matched_col_inds]
        # pos_inds = matched_row_inds[(check_parts.max(1)[0].bool() * (ck > 0.5))[matched_col_inds]]
        if is_weighted:
            flag = affinity_targets.sum(-1, keepdim=True)
            flag = flag.masked_fill(flag == 0, 1)
            affinity_weights = affinity_weights / flag

        bbox_targets = affinity_pred.new_zeros((num_quries, 4))
        pos_gt_bboxes_normalized = gt_instances.bboxes[matched_col_inds] / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        # if matched_row_inds.max()>num_patches:
        #     print(matched_row_inds)
        return (labels, label_weights, affinity_targets, bbox_targets,
                affinity_weights, pos_inds, neg_inds, ignore_flag)

    def get_bboxes_segms(self, affinity_pred, proposals, cls_score, img_shape, norm=False, thre=0.5,
                         return_valid=False, is_patch_query=False):
        proposal_boxes = proposals.bboxes.tensor
        affinity_pred = affinity_pred.sigmoid()
        if norm:
            affinity_pred = (affinity_pred - affinity_pred.min(-1)[0][:, None]) / (
                    affinity_pred.max(-1)[0][:, None] - affinity_pred.min(-1)[0][:, None])
        if self.train_cfg.get('with_affinity_only_highest', False):
            affinity_pred = affinity_pred >= affinity_pred.max(-1)[0][:, None]

        if is_patch_query:
            num_patches = len(affinity_pred)
            affinity_pred[torch.arange(num_patches), torch.arange(num_patches)] = 1

        aa = ((affinity_pred[..., None] > thre) * proposal_boxes[None, ...])
        ignorezero = (affinity_pred > thre) == 0
        aa[ignorezero, :] = cls_score.new([float('inf'), float('inf'), float('-inf'), float('-inf')])
        x1y1 = aa[..., :2].min(1)[0]
        x2y2 = aa[..., 2:].max(1)[0]
        bboxes_pred = torch.cat([x1y1, x2y2], dim=-1)
        ignorezero_b = (affinity_pred > thre).max(-1)[0] == 0
        bboxes_pred[ignorezero_b] = cls_score.new([float(0), float(0), float(1), float(1)])

        if 'masks' in proposals:
            for_merge = (affinity_pred > thre)
            masks = torch.tensor(proposals.masks.masks, device=affinity_pred.device)
            batch_size = 10
            batch_for_merge = torch.split(for_merge, batch_size)
            mask_pred = []
            for fm in batch_for_merge:
                batch_mask_pred = (fm[..., None, None] * masks[None, ...]).max(1)[0]
                mask_pred.append(batch_mask_pred)
            mask_pred = torch.cat(mask_pred)

            if return_valid:
                return bboxes_pred, mask_pred, affinity_pred, ignorezero_b
            else:
                return bboxes_pred, mask_pred, affinity_pred
        else:
            if return_valid:
                return bboxes_pred, None, affinity_pred, ignorezero_b
            else:
                return bboxes_pred, None, affinity_pred

    def predict(self,
                hidden_states: Tensor,
                references: Tensor,
                batch_data_samples: SampleList,
                affinities: Tensor,
                patch_fts: Tensor,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_proposls = [data_samples.proposals for data_samples in batch_data_samples]
        all_layers_outputs_classes, references, patch_outputs_classes = self(hidden_states,
                                                                             patch_fts if self.use_patch_cls else None,
                                                                             references)

        predictions = self.predict_by_feat(
            all_layers_outputs_classes, affinities, references, batch_proposls=batch_proposls,
            batch_img_metas=batch_img_metas, batch_data_samples=batch_data_samples,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_affinities: Tensor,
                        all_layers_coors: Tensor,
                        batch_img_metas: List[Dict],
                        batch_data_samples,
                        batch_proposls: List[Dict],
                        rescale: bool = False) -> InstanceList:
        panoptic_on = self.test_cfg.get('panoptic_on', False)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', True)
        cls_scores = all_layers_cls_scores[-1]
        if self.use_regression:
            cls_scores[:, :self.num_stuff_classes] = all_layers_cls_scores[-2][:, :self.num_stuff_classes]
        affinities = all_layers_affinities[-2] if self.use_regression else all_layers_affinities[-1]
        coors = all_layers_coors[-1]
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, coors):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.stack(factors, 0)
        coors = bbox_cxcywh_to_xyxy(coors) * factors
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            affinity = affinities[img_id]
            coor = coors[img_id]
            img_meta = batch_img_metas[img_id]
            proposal = batch_proposls[img_id]
            results = dict()
            if instance_on:
                if self.semantic_cfg is not None:
                    ins_results = self._predict_by_feat_single(
                        cls_score[self.num_stuff_classes:, :-self.num_stuff_classes],
                        affinity[self.num_stuff_classes:], coor[self.num_stuff_classes:],
                        img_meta, proposal, rescale, batch_data_samples[img_id])
                else:
                    ins_results = self._predict_by_feat_single(cls_score, affinity, coor,
                                                               img_meta, proposal, rescale, batch_data_samples[img_id])
                results['ins_results'] = ins_results
            if panoptic_on:
                pan_results, sem_results = self._predict_by_feat_panoptic_single(cls_score, affinity, coor,
                                                                                 img_meta, proposal, rescale,
                                                                                 batch_data_samples[img_id])
                results['pan_results'] = pan_results
                results['sem_results'] = sem_results
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                affinity: Tensor,
                                coor: Tensor,
                                img_meta: dict,
                                proposal: dict,
                                rescale: bool = True,
                                data_samples=None) -> InstanceData:
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        with_score_norm = self.train_cfg.get('with_score_norm', False)
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.3)
        img_shape = img_meta['img_shape']
        num_patches = proposal.bboxes.shape[0]
        affinity = affinity[:, :num_patches]
        if self.use_patch_query:
            affinity = affinity[:num_patches]
            # if self.
            cls_score = cls_score[:num_patches]
            coor = coor[:num_patches]
        # exclude background
        if self.loss_cls.use_sigmoid:

            use_nms = self.test_cfg.get('with_nms', False)
            if use_nms:

                bbox_pred, mask_pred, _ = self.get_bboxes_segms(affinity, proposal, cls_score.sigmoid(), img_shape,
                                                                norm=with_score_norm,
                                                                thre=0.5)

                ins_scores = cls_score.sigmoid()
                det_bboxes, det_labels, index = multiclass_nms(
                    bbox_pred,
                    torch.cat([ins_scores, ins_scores.new_zeros(len(ins_scores), 1)], dim=1),
                    0.05,
                    dict(type='nms', iou_threshold=0.8),
                    100,
                    return_inds=True,
                    box_dim=4)
                keep_idxs = index // self.num_things_classes
                scores = det_bboxes[:, -1]
                bbox_pred = det_bboxes[:, :4]
                mask_pred = mask_pred[keep_idxs]





            else:
                cls_score = cls_score.sigmoid()
                num_classes = cls_score.shape[-1]
                scores, indexes = cls_score.view(-1).topk(max_per_img)
                det_labels = indexes % num_classes
                bbox_index = indexes // num_classes
                affinity_pred = affinity[bbox_index]

                bbox_pred, mask_pred, _ = self.get_bboxes_segms(affinity_pred, proposal, scores, img_shape,
                                                                norm=with_score_norm,
                                                                thre=0.5)
                bbox_pred = coor[bbox_index] if self.use_regression else bbox_pred

            det_bboxes = bbox_pred

        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        if mask_pred is not None:
            if proposal['proposals_segm_mode'][0] == 'OriImage':
                mask_pred_result = mask_pred.detach().cpu().numpy()

            else:
                img_height, img_width = img_meta['img_shape'][:2]
                mask_pred_result = mask_pred[:, :img_height, :img_width]
                mask_pred_result = mask_pred_result.detach().cpu().numpy()
                # return result in original resolution
                ori_height, ori_width = img_meta['ori_shape'][:2]
                mask_pred_result = mmcv.imresize(mask_pred_result.transpose(1, 2, 0), (ori_width, ori_height))
                if len(mask_pred_result.shape) == 2:
                    mask_pred_result = mask_pred_result[..., None]
                mask_pred_result = mask_pred_result.transpose(2, 0, 1)

        results = InstanceData()

        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        if mask_pred is not None:
            results.masks = det_bboxes.new_tensor(mask_pred_result).bool()
        return results

    def _predict_by_feat_panoptic_single(self,
                                         cls_score: Tensor,
                                         affinity: Tensor,
                                         coor: Tensor,
                                         img_meta: dict,
                                         proposal: dict,
                                         rescale: bool = True,
                                         data_samples=None) -> InstanceData:
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        with_score_norm = self.train_cfg.get('with_score_norm', False)
        img_shape = img_meta['img_shape']
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.3)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)
        num_patches = proposal.bboxes.shape[0]
        affinity = affinity[:, :num_patches]
        if self.use_patch_query:
            affinity = affinity[:num_patches + self.num_stuff_classes]
            cls_score = cls_score[:num_patches + self.num_stuff_classes]
            coor = coor[:num_patches + self.num_stuff_classes]
        # exclude background
        if self.loss_cls.use_sigmoid:
            use_nms = self.test_cfg.get('with_nms', False)
            if use_nms:

                bbox_pred, mask_pred, _ = self.get_bboxes_segms(affinity, proposal, cls_score.sigmoid(), img_shape,
                                                                norm=with_score_norm,
                                                                thre=0.5, is_patch_query=False)

                ins_scores = cls_score[self.num_stuff_classes:, :-self.num_stuff_classes].sigmoid()
                ins_bbox, ins_mask = bbox_pred[self.num_stuff_classes:], mask_pred[self.num_stuff_classes:]
                det_bboxes, det_labels, index = multiclass_nms(
                    ins_bbox,
                    torch.cat([ins_scores, ins_scores.new_zeros(len(ins_scores), 1)], dim=1),
                    0.05,
                    dict(type='nms', iou_threshold=0.8),
                    100,
                    return_inds=True,
                    box_dim=4)
                keep_idxs = index // self.num_things_classes
                det_scores = det_bboxes[:, -1]
                bbox_pred = det_bboxes[:, :4]
                ins_mask = ins_mask[keep_idxs]

                sem_mask = mask_pred[:self.num_stuff_classes]
                semantic_scores, sematic_lables = cls_score.sigmoid()[torch.arange(self.num_stuff_classes),
                self.num_things_classes + torch.arange(self.num_stuff_classes)], self.num_things_classes + torch.arange(
                    self.num_stuff_classes)
                sematic_lables = sematic_lables.to(semantic_scores.device)
                sematic_lables = sematic_lables[semantic_scores > object_mask_thr]
                sem_mask = sem_mask[semantic_scores > object_mask_thr]
                semantic_scores = semantic_scores[semantic_scores > object_mask_thr]
                scores = torch.cat([semantic_scores, det_scores])
                labels = torch.cat([sematic_lables, det_labels])
                mask_pred = torch.cat([sem_mask, ins_mask])

                labels = labels[scores > object_mask_thr]
                # bbox_pred=bbox_pred[scores>object_mask_thr]
                mask_pred = mask_pred[scores > object_mask_thr]
                scores = scores[scores > object_mask_thr]
            else:
                #
                scores, labels = cls_score[self.num_stuff_classes:, :-self.num_stuff_classes].sigmoid().max(-1)
                semantic_scores, sematic_lables = cls_score.sigmoid()[torch.arange(self.num_stuff_classes),
                self.num_things_classes + torch.arange(self.num_stuff_classes)], self.num_things_classes + torch.arange(
                    self.num_stuff_classes)

                sematic_lables = sematic_lables.to(semantic_scores.device)
                scores = torch.cat([semantic_scores, scores])
                labels = torch.cat([sematic_lables, labels])
                bbox_index = (scores > object_mask_thr) * (affinity.sigmoid().max(-1)[0] > 0.3)
                tpk = scores.topk(min(max_per_img, len(scores)))[1]
                bbox_index = bbox_index[tpk]
                scores = scores[tpk][bbox_index]
                labels = labels[tpk][bbox_index]
                affinity_pred = affinity[tpk][bbox_index]

                bbox_pred, mask_pred, _ = self.get_bboxes_segms(affinity_pred, proposal, scores, img_shape,
                                                                norm=with_score_norm,
                                                                thre=0.5, is_patch_query=False)
                # semantic_scores, sematic_lables = cls_score.sigmoid()[torch.arange(self.num_stuff_classes),
                # self.num_things_classes + torch.arange(
                #     self.num_stuff_classes)], self.num_things_classes + torch.arange(
                #     self.num_stuff_classes)
                #
                # instance_scores = cls_score[self.num_stuff_classes:, :-self.num_stuff_classes].sigmoid()
                #
                # scores, indexes = instance_scores.view(-1).topk(max_per_img)
                # labels = indexes % self.num_things_classes
                # bbox_index = indexes // self.num_things_classes
                # affinity_pred = affinity[self.num_stuff_classes:][bbox_index]
                # bbox_pred, mask_pred, _ = self.get_bboxes_segms(affinity_pred, proposal, scores, img_shape,
                #                                                 norm=with_score_norm,
                #                                                 thre=0.5)
                #
                # sematic_lables = sematic_lables.to(semantic_scores.device)
                # # scores = torch.cat([semantic_scores, scores])
                #
                # ##semantic
                # _, sem_mask_pred, _ = self.get_bboxes_segms(affinity[:self.num_stuff_classes], proposal,
                #                                             semantic_scores, img_shape,
                #                                             norm=True,
                #                                             thre=0.5)
                # sem_mask_pred = sem_mask_pred[semantic_scores > object_mask_thr]
                # sematic_lables = sematic_lables[semantic_scores > object_mask_thr]
                # semantic_scores = semantic_scores[semantic_scores > object_mask_thr]
                # mask_pred = torch.cat([sem_mask_pred, mask_pred], dim=0)
                # scores = torch.cat([semantic_scores, scores])
                # labels = torch.cat([sematic_lables, labels])
                # # mask_pred = mask_pred[scores > object_mask_thr]
                # # labels = labels[scores > object_mask_thr]
                # # scores = scores[scores > object_mask_thr]

            if mask_pred is not None:
                if proposal['proposals_segm_mode'][0] == 'OriImage':
                    mask_pred_result = mask_pred.detach().cpu().numpy()
                else:
                    img_height, img_width = img_meta['img_shape'][:2]
                    mask_pred_result = mask_pred[:, :img_height, :img_width]
                    ori_height, ori_width = img_meta['ori_shape'][:2]
                    if mask_pred_result.shape[0] > 0:
                        mask_pred_result = mask_pred_result.detach().cpu().numpy()
                        # return result in original resolution
                        mask_pred_result = mmcv.imresize(mask_pred_result.transpose(1, 2, 0), (ori_width, ori_height))

                        if len(mask_pred_result.shape) == 2:
                            mask_pred_result = mask_pred_result[..., None]
                        mask_pred_result = mask_pred_result.transpose(2, 0, 1)
                    else:
                        mask_pred_result = mask_pred.new_zeros((0, ori_height, ori_width))
        if self.semantic_cfg is not None:
            mask_pred_result = affinity.new_tensor(mask_pred_result)

            cur_masks = mask_pred_result
            cur_scores = scores
            cur_classes = labels
            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.full((h, w),
                                      self.num_classes,
                                      dtype=torch.int32,
                                      device=cur_masks.device)
            semntic_seg = torch.full((h, w),
                                     self.num_classes,
                                     dtype=torch.int32,
                                     device=cur_masks.device)
            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                pass
            else:
                cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                        semntic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = (
                                pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1
                        semntic_seg[mask] = pred_class
            panotic_results = PixelData(sem_seg=panoptic_seg[None])
            semntic_results = PixelData(sem_seg=semntic_seg[None])
        return panotic_results, semntic_results
