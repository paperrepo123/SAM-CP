# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class PartAssignerPlus(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        iou_calculator (dict): Config of overlaps Calculator.
    """

    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: Union[float, tuple],
                 min_pos_iou: float = .0,
                 gt_max_assign_all: bool = True,
                 ignore_iof_thr: float = -1,
                 ignore_wrt_candidates: bool = True,
                 match_low_quality: bool = True,
                 match_high_iou_part: bool = True,
                 low_quality_add_segm=False,
                 gpu_assign_thr: float = -1,
                 iou_calculator: dict = dict(type='BboxOverlaps2D'),
                 offline_assign=dict(
                     offline_overlap='/cache/pretrained/sam_proposals/coco_train/overlaps/',
                     proposal_mode='segm',
                     calculate_mode='iof', )
                 ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.match_high_iou_part = match_high_iou_part
        self.low_quality_add_segm=low_quality_add_segm
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        if offline_assign is not None:
            self.calculate_mode = offline_assign['calculate_mode']
            self.proposal_mode = offline_assign['proposal_mode']
            self.offline_overlap = offline_assign['offline_overlap']

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:

        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.bboxes
        gt_labels = gt_instances.labels
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
                gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

        if not self.offline_overlap:
            from .calculate_mask_overlap import calcualte_overlap
            offline_overlaps = calcualte_overlap(pred_instances, gt_instances, mode=self.proposal_mode,
                                                 calculate_mode=self.calculate_mode)
            if isinstance(self.proposal_mode, list):
                overlaps = [offline_overlaps[md][self.calculate_mode] for md in self.proposal_mode]
                overlaps = torch.stack(overlaps).min(0)[0]
            else:
                overlaps = offline_overlaps[self.proposal_mode][self.calculate_mode]
            overlaps_for_low_quality = self.iou_calculator(gt_bboxes, priors)
        # else:
        else:
            offline_overlaps = torch.load(
                self.offline_overlap + img_meta['img_path'].split('/')[-1].split('.')[0] + '.pkl',
                map_location=priors.device)

            valid_gt = gt_instances['gt_valid_ids'] if 'gt_valid_ids' in gt_instances else torch.arange(len(gt_bboxes))

            valid_pps = pred_instances[
                'proposals_valid_ids'] if 'proposals_valid_ids' in pred_instances else torch.arange(len(priors))

            if isinstance(self.proposal_mode, list):
                overlaps = [offline_overlaps[md][self.calculate_mode][valid_gt][:, valid_pps] for md in
                            self.proposal_mode]
                overlaps = torch.stack(overlaps).min(0)[0]
            else:
                overlaps = offline_overlaps[self.proposal_mode][self.calculate_mode][valid_gt][:, valid_pps]
            if self.low_quality_add_segm:
                overlaps_for_low_quality = [offline_overlaps[md]['iou'][valid_gt][:, valid_pps] for md in
                                            self.proposal_mode]
                overlaps_for_low_quality = torch.stack(overlaps_for_low_quality).min(0)[0]
            else:
            # else:
                overlaps_for_low_quality = offline_overlaps['bbox']['iou'][valid_gt][:, valid_pps]

        # overlaps = self.iou_calculator(gt_bboxes, priors)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    priors, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, priors, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, overlaps_for_low_quality, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps: Tensor, overlaps_for_low_quality: Tensor,
                            gt_labels: Tensor) -> AssignResult:
        """Assign w.r.t. the overlaps of priors with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, num_gts),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, num_gts))
            assigned_labels = None
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps = overlaps.permute(1, 0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = 1
        overlaps_for_low_quality = overlaps_for_low_quality.permute(1, 0)
        if self.match_high_iou_part:
            # assigned_gt_inds[(
            #         overlaps_for_low_quality > (overlaps_for_low_quality * assigned_gt_inds.permute(1, 0)).max(-1)[
            #                                        0][:, None])] = 1
            # print(assigned_gt_inds[(overlaps_for_low_quality > (overlaps_for_low_quality * assigned_gt_inds).max(
            #     0)[0][None, :]) * (overlaps_for_low_quality > 0.5)])
            assigned_gt_inds[(overlaps_for_low_quality > (overlaps_for_low_quality * assigned_gt_inds).max(
                0)[0][None, :]) * (overlaps_for_low_quality > 0.5)] = 1

        if self.match_low_quality:

            is_assigned = ~assigned_gt_inds.max(0)[0].bool()
            # print(is_assigned)
            assigned_gt_inds[overlaps_for_low_quality.argmax(0)[is_assigned],
            torch.arange(overlaps_for_low_quality.shape[1], device=max_overlaps.device)[is_assigned]] = (
                    overlaps_for_low_quality[overlaps_for_low_quality.argmax(0)[
                        is_assigned], torch.arange(overlaps_for_low_quality.shape[1], device=max_overlaps.device)[
                        is_assigned]] > self.min_pos_iou).long()
        # if (~assigned_gt_inds.max(0)[0].bool()==is_assigned).min()==False:
        #     print(~assigned_gt_inds.max(0)[0].bool())
        assigned_labels = None
        # print(assigned_gt_inds.sum(1).max())
        # if assigned_gt_inds.sum(1).max()>1:
        #     print('aa')
        #     pass
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)
