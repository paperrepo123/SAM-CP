
import json

import torch
import numpy as np
import pickle
from pycocotools.coco import COCO
# from segment_anything_main.test.offline_test import _parse_ann_info
from pycocotools import mask as maskUtils
from mmdet.structures.bbox import bbox_overlaps
import os
import json
from tqdm import tqdm
# from multiprocplus import multiprocess_for, MultiprocessRunner
import argparse


def _parse_ann_info(img_info, ann_info, coco):
    """Parse bbox and mask annotation.

    Args:
        ann_info (list[dict]): Annotation info of an image.
        with_mask (bool): Whether to parse mask annotations.

    Returns:
        dict: A dict containing the following keys: bboxes, bboxes_ignore,\
            labels, masks, seg_map. "masks" are raw annotations and not \
            decoded into binary masks.
    """
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []
    gt_areas = []
    gt_scores = []
    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
        if inter_w * inter_h == 0:
            continue
        if ann['area'] <= 0 or w < 1 or h < 1:
            continue
        if ann['category_id'] not in coco.getCatIds():
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            cat2label = {cat_id: i for i, cat_id in enumerate(coco.getCatIds())}
            gt_labels.append(cat2label[ann['category_id']])
            gt_masks_ann.append(ann.get('segmentation', None))
            gt_areas.append(ann['area'])
            if ann.get('score'):
                gt_scores.append(ann['score'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        gt_areas = np.array(gt_areas, dtype=np.float32)
        if ann.get('score'):
            gt_scores = np.array(gt_scores, dtype=np.float32)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)
        gt_areas = np.zeros((0), dtype=np.float32)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    seg_map = img_info['file_name'].replace('jpg', 'png')

    ann = dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        bboxes_ignore=gt_bboxes_ignore,
        masks=gt_masks_ann,
        areas=gt_areas,
        seg_map=seg_map,
    )
    if len(gt_scores) > 0:
        ann.update(scores=gt_scores)

    return ann


def annToRLE(segm, img_size):
    h, w = img_size
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    return rle


def annToMask(segm, img_size):
    if type(segm).__name__ != 'dict':
        rle = annToRLE(segm, img_size)
    else:
        rle = segm
    m = maskUtils.decode(rle)
    return m


def mask_overlaps(m1, m2, mode='iou'):
    overlap = ((m1[:, None, ...] + m2[None, ...]).flatten(2) == 2).sum(-1)
    if mode == 'iou':
        union = ((m1[:, None, ...] + m2[None, ...]).flatten(2) >= 1).sum(-1)
    elif mode == 'iog':
        union = (m1[:, None, ...]).flatten(2).sum(-1)
    elif mode == 'iof':
        union = (m2[None, :, ...]).flatten(2).sum(-1)
    iou_mask = overlap / (union+1e-10)

    return iou_mask


from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

def batch_iterator(batch_size: int, *args) :
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def calcualte_overlap(sam_result, gt_instance, mode=['bbox', 'segm'],
                      calculate_mode=['iou', 'iof', 'iog']):
    gt_bboxes = gt_instance.bboxes
    if len(gt_bboxes) > 0:
        gt_masks = gt_instance.masks.resize(
            (round(gt_instance.masks.height / 4), round(gt_instance.masks.width / 4))).masks
        pred_proposals_boxes = sam_result.bboxes.tensor
        pred_proposals_masks = sam_result.masks.resize(
            (round(sam_result.masks.height / 4), round(sam_result.masks.width / 4))).masks

        overlaps = dict()
        if 'bbox' in mode:
            overlaps['bbox'] = dict()
            overlaps['bbox']['iou'] = bbox_overlaps(gt_bboxes, pred_proposals_boxes,
                                                    mode='iou') if 'iou' in calculate_mode else None
            overlaps['bbox']['iof'] = bbox_overlaps(pred_proposals_boxes, gt_bboxes, mode='iof').permute(1,
                                                                                                         0) if 'iof' in calculate_mode else None
            overlaps['bbox']['iog'] = bbox_overlaps(gt_bboxes, pred_proposals_boxes,
                                                    mode='iof') if 'iog' in calculate_mode else None
        if 'segm' in mode:
            overlaps['segm'] = dict()
            batch_size = 100
            p_batch_size = 1000
            # batch_gt_masks = torch.split(gt_masks, batch_size)
            # batch_pred_proposals_masks = torch.split(pred_proposals_masks, p_batch_size)
            iou = []
            iof = []
            iog = []
            for i in range(0, len(gt_masks), batch_size):
                # for mask in batch_gt_masks:
                iou_ = []
                iof_ = []
                iog_ = []
                for j in range(0, len(pred_proposals_masks), p_batch_size):
                    # for p_mask in batch_pred_proposals_masks:
                    mask = gt_bboxes.new_tensor(gt_masks[i: i + batch_size])
                    p_mask = gt_bboxes.new_tensor(pred_proposals_masks[j:j + p_batch_size])
                    if mask.shape[0] > 0 and p_mask.shape[0] > 0:
                        if 'iou' in calculate_mode:
                            iou_.append(mask_overlaps(mask, p_mask, mode='iou'))
                        if 'iof' in calculate_mode:
                            iof_.append(mask_overlaps(mask, p_mask, mode='iof'))
                        if 'iog' in calculate_mode:
                            iog_.append(mask_overlaps(mask, p_mask, mode='iog'))
                if 'iou' in calculate_mode:
                    iou.append(torch.cat(iou_, dim=-1))
                if 'iof' in calculate_mode:
                    iof.append(torch.cat(iof_, dim=-1))
                if 'iog' in calculate_mode:
                    iog.append(torch.cat(iog_, dim=-1))

            overlaps['segm']['iou'] = torch.cat(iou) if 'iou' in calculate_mode else None
            overlaps['segm']['iof'] = torch.cat(iof) if 'iof' in calculate_mode else None
            overlaps['segm']['iog'] = torch.cat(iog) if 'iog' in calculate_mode else None
        # delete

    return overlaps


#
# def calcualte_overlap(sam_results, coco, output_dict, mode=['bbox', 'segm'],
#                       calculate_mode=['iou', 'iof', 'iog']):
#     # assert mode in ['bbox', 'segm']
#     # assert calculate_mode in ['iou', 'iof', 'iog']
#     if not os.path.isdir(output_dict):
#         os.mkdir(output_dict)
#     multiprocess_for(bbox_mask_overlap, sam_results, coco,
#                      [(annIds_list_split, num) for annIds_list_split in annIds_list_split],
#                      num_process=22)
#     for i, img_id in tqdm(enumerate(sam_results['indexes']), total=len(sam_results['indexes'])):
#
#     return True
# iou = bbox_overlaps(pred_proposals, gt_bboxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    args = parser.parse_args()
    dataset_type = 'train'
    mode = ['bbox', 'segm']
    calculate_mode = ['iou', 'iof', 'iog']
    sam_results_file = f'/cache/pretrained/sam_proposals/coco_{dataset_type}/sam_proposal_vit-h_{dataset_type}.pkl'
    sam_results = pickle.load(open(sam_results_file, 'rb'))
    gt_annotations_file = f'/cache/coco/annotations/instances_{dataset_type}2017.json'
    coco = COCO(gt_annotations_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dict = f'/cache/pretrained/sam_proposals/coco_{dataset_type}/overlaps/'
    output_proposal_dict = f'/cache/pretrained/sam_proposals/coco_{dataset_type}/proposals_vit-h_{dataset_type}/'

    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)
    if not os.path.isdir(output_proposal_dict):
        os.mkdir(output_proposal_dict)

    extract_sam_results = []
    extract_gt_instances = []
    args.start = 0 if args.start is None else int(args.start)
    args.end = len(sam_results['indexes']) if args.end is None else int(args.end)
    for i, img_id in enumerate(tqdm(sam_results['indexes'])):
        if args.start <= i < args.end:
            per_img = dict()
            for j in sam_results.keys():
                per_img[j] = sam_results[j][i]
            extract_sam_results.append(per_img)
            gt_anns_info = coco.loadAnns(coco.getAnnIds(img_id))
            gt_img_info = coco.loadImgs(img_id)
            gt_ann = _parse_ann_info(gt_img_info[0], gt_anns_info, coco)
            gt_img_info = coco.loadImgs(img_id)[0]
            extract_gt_instances.append([gt_img_info, gt_ann])

            f_name = '0' * (12 - len(str(img_id))) + str(img_id)
            output_proposal_file = output_proposal_dict + f_name + '.pkl'
            pickle.dump(per_img, open(output_proposal_file, 'wb'))
    print('start calculate overlap:')
    # a = multiprocess_for(calcualte_overlap,
    #                      [(sam_result, gt_instance) for sam_result, gt_instance in
    #                       tqdm(zip(extract_sam_results, extract_gt_instances),total=len(extract_sam_results))],
    #                      num_process=22)
    for sam_result, gt_instance in tqdm(zip(extract_sam_results, extract_gt_instances),
                                        total=len(extract_gt_instances)):
        calcualte_overlap(sam_result, gt_instance)

    # for
    # overlaps = calcualte_overlap(sam_results['indexes'][0], sam_results, coco, output_dict)