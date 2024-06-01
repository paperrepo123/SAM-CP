# import os
# import json
# import math
# import argparse
# from tqdm import tqdm
#
# import numpy as np
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
#
# from mmengine.runner import Runner
#
# from mmdet.registry import DATASETS
# from mmdet.utils import register_all_modules
# from mmdet.structures.mask import BitmapMasks
#
# from segment_anything_main.segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
# from segment_anything_main.segment_anything.utils.amg import mask_to_rle_pytorch, coco_encode_rle
# from mmdet.structures.bbox import scale_boxes
from mmdet.utils import register_all_modules
import json
import math
import argparse
from tqdm import tqdm
import pickle
from mmengine.runner import Runner
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from extra_tools.transform import x1y1wh_to_xyxy
# from mmengine.runner import Runner
from mmdet.registry import DATASETS
# from mmdet.datasets import build_dataset, build_dataloader
# from mmdet.utils import register_all_modules
from mmdet.structures.mask import BitmapMasks

from segment_anything_main.segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything_main.segment_anything.utils.amg import mask_to_rle_pytorch, coco_encode_rle
from mmdet.structures.bbox import scale_boxes
from pycocotools.coco import COCO
import glob

# from segment_anything_main.test.offline_test import _parse_ann_info
from pycocotools import mask as maskUtils
from mmdet.structures.bbox import bbox_overlaps
import os
from multiprocplus import multiprocess_for, MultiprocessRunner
import argparse
from mmengine.structures import InstanceData, PixelData
from mmdet.models.utils import multi_apply


def data_builder(cfg):
    dataset_type = cfg['type']
    data_root = cfg['data_root']
    ann_file = cfg['ann_file']
    data_prefix = cfg['data_prefix']

    backend_args = None
    pipeline = [
        dict(type='LoadImageFromFile', backend_args=backend_args),
        dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
        dict(
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True,
            with_seg=True,
            reduce_zero_label=True),
        dict(
            type='PackDetInputs', meta_keys=('img_path', 'ori_shape', 'img_shape',
                                             'scale_factor'))
    ]

    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=data_prefix,
        test_mode=True,
        pipeline=pipeline,
        backend_args=backend_args)

    dataloader_cfg = dict(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        persistent_workers=cfg['num_workers'] > 0,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dataset)

    dataset = DATASETS.build(dataset)
    data_loader = Runner.build_dataloader(dataloader_cfg)

    return data_loader, dataset


def dataset_builder(cfg):
    dataset_type = cfg['dataset_type']
    data_root = cfg['data_root']
    ann_file = cfg['ann_file']

    backend_args = None
    pipeline = [
        dict(type='LoadImageFromFile', backend_args=backend_args),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        # dict(type='Pad', size=(1333, 800), pad_val=dict(img=(114, 114, 114))),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor'))
    ]

    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img=cfg.get('data_prefix', '')),
        test_mode=True,
        pipeline=pipeline,
        backend_args=backend_args)

    dataset = DATASETS.build(dataset)

    return dataset


def data_loader_builder(dataset, cfg, num_data):
    dataset = [dataset[-(i + 1)] for i in range(num_data)]

    dataloader_cfg = dict(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        persistent_workers=cfg['num_workers'] > 0,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dataset)

    data_loader = Runner.build_dataloader(dataloader_cfg)

    return data_loader


def box_xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    from copy import deepcopy
    box_xyxy = deepcopy(box_xywh)
    box_xyxy[:, 2] = box_xyxy[:, 2] + box_xyxy[:, 0]
    box_xyxy[:, 3] = box_xyxy[:, 3] + box_xyxy[:, 1]
    return box_xyxy


def transfer_samresults_to_proposalfile(results_file_list):
    proposal_dict = dict()
    indexes = []
    boxes = []
    segmentations = []
    scores = []
    for i in tqdm(range(len(results_file_list))):
        ## add results to proposal per image
        results = json.load(open(results_file_list[i]))
        indexes.append(results[0]['image_id'])
        bbx = [r['bbox'] for r in results]
        boxes.append(np.array(bbx, dtype=np.float32))
        segm = [r['segmentation'] for r in results]
        segmentations.append(segm)
        h, w = coco.loadImgs(results[0]['image_id'])[0]['height'], coco.loadImgs(results[0]['image_id'])[0]['width']
        # shape=cv2.imread('/cache/coco/train2017/'+'0'*(12-len(str(results[0]['image_id'])))+str(results[0]['image_id'])+'.jpg').shape
        if h != segm[0]['size'][0] or w != segm[0]['size'][1]:
            for j in segm:
                j['size'][0] = h
                j['size'][1] = w
            print((h, w), segm[0]['size'])
        sco = [r['score'] for r in results]
        scores.append(np.array(sco, dtype=np.float32)[:, None])
    proposal_dict.update({'indexes': indexes, 'boxes': boxes, 'segmentations': segmentations, 'scores': scores})
    return proposal_dict


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
    iou_mask = overlap / union

    return iou_mask


def calcualte_overlap(sam_result, gt_instance, mode=['bbox', 'segm'],
                      calculate_mode=['iou', 'iof', 'iog'], output_dict=None):
    img_id = sam_result['indexes']
    f_name = '0' * (12 - len(str(img_id))) + str(img_id)
    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)
    output_file = output_dict + f_name + '.pkl'
    # if os.path.isfile(output_file):
    #     return True

    gt_img_info, gt_anns_info = gt_instance
    if 'bbox' in mode:
        gt_bboxes = torch.tensor(gt_anns_info['bboxes']).to(device)
        len_ann = len(gt_bboxes)
    if 'segm' in mode:
        # gt_masks = np.array([annToMask(mk, (gt_img_info['height'],
        #                                     gt_img_info['width'])) for mk in gt_anns_info['masks']])
        gt_masks = gt_anns_info['masks']
        gt_masks = torch.tensor(gt_masks).to(device)
        len_ann = len(gt_masks)

    if len_ann > 0:
        pred_proposals_boxes = torch.tensor(x1y1wh_to_xyxy(sam_result['boxes'])).to(device)
        pred_proposals_masks = np.array([annToMask(mk, (gt_img_info['img_shape'][0],
                                                        gt_img_info['img_shape'][1])) for mk in
                                         sam_result['segmentations']])
        pred_proposals_masks = torch.tensor(pred_proposals_masks).to(device)

        overlaps = dict()
        if 'bbox' in mode:
            overlaps['bbox'] = dict()
            if 'iou' in calculate_mode:
                overlaps['bbox']['iou'] = bbox_overlaps(gt_bboxes, pred_proposals_boxes,
                                                        mode='iou') if 'iou' in calculate_mode else None
                overlaps['bbox']['iof'] = bbox_overlaps(pred_proposals_boxes, gt_bboxes, mode='iof').permute(1,
                                                                                                             0) if 'iof' in calculate_mode else None
                overlaps['bbox']['iog'] = bbox_overlaps(gt_bboxes, pred_proposals_boxes,
                                                        mode='iof') if 'iog' in calculate_mode else None
        if 'segm' in mode:
            overlaps['segm'] = dict()
            batch_size = 10
            p_batch_size = 300
            batch_gt_masks = torch.split(gt_masks, batch_size)
            batch_pred_proposals_masks = torch.split(pred_proposals_masks, p_batch_size)
            iou = []
            iof = []
            iog = []
            for mask in batch_gt_masks:
                iou_ = []
                iof_ = []
                iog_ = []
                for p_mask in batch_pred_proposals_masks:
                    if mask.shape[0] > 0 and p_mask.shape[0] > 0:
                        if 'iou' in calculate_mode:
                            iou_.append(mask_overlaps(mask, p_mask, mode='iou'))
                        if 'iof' in calculate_mode:
                            iof_.append(mask_overlaps(mask, p_mask, mode='iof'))
                        if 'iog' in calculate_mode:
                            iog_.append(mask_overlaps(mask, p_mask, mode='iog'))
                iou.append(torch.cat(iou_, dim=-1))
                iof.append(torch.cat(iof_, dim=-1))
                iog.append(torch.cat(iog_, dim=-1))

            iou = torch.cat(iou)
            iof = torch.cat(iof)
            iog = torch.cat(iog)
            overlaps['segm']['iou'] = iou if 'iou' in calculate_mode else None
            overlaps['segm']['iof'] = iof if 'iof' in calculate_mode else None
            overlaps['segm']['iog'] = iog if 'iog' in calculate_mode else None
        if 'labels' in gt_anns_info:
            overlaps['labels'] = gt_anns_info['labels']
        torch.save(overlaps, output_file)
    return True



def preprocess_gt(
        batch_gt_instances,
        batch_gt_semantic_segs):
    """Preprocess the ground truth for all images.

    Args:
        batch_gt_instances (list[:obj:`InstanceData`]): Batch of
            gt_instance. It usually includes ``labels``, each is
            ground truth labels of each bbox, with shape (num_gts, )
            and ``masks``, each is ground truth masks of each instances
            of a image, shape (num_gts, h, w).
        gt_semantic_seg (list[Optional[PixelData]]): Ground truth of
            semantic segmentation, each with the shape (1, h, w).
            [0, num_thing_class - 1] means things,
            [num_thing_class, num_class-1] means stuff,
            255 means VOID. It's None when training instance segmentation.

    Returns:
        list[obj:`InstanceData`]: each contains the following keys

            - labels (Tensor): Ground truth class indices\
                for a image, with shape (n, ), n is the sum of\
                number of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a\
                image, with shape (n, h, w).
    """
    num_things_classes, num_stuff_classes = 100, 50
    num_things_list = [num_things_classes] * len(batch_gt_instances)
    num_stuff_list = [num_stuff_classes] * len(batch_gt_instances)
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

    targets = multi_apply(preprocess_semantic_gt, gt_semantic_segs, num_things_list,
                          num_stuff_list)
    masks, labels = targets
    batch_gt_instances = [
        InstanceData(labels=label, masks=mask)
        for label, mask in zip(labels, masks)
    ]
    return batch_gt_instances


def preprocess_semantic_gt( gt_semantic_seg, num_things,
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


if __name__ == '__main__':
    register_all_modules()
    dataset_settype = 'val'
    if dataset_settype == 'validation':
        type_ = 'val'
    elif dataset_settype == 'training':
        type_ = 'train'
    cfg = dict(
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        type='CityscapesDataset',
        data_root='/home/cpf/dataset/cityscapes/',
        ann_file=f'annotations/instancesonly_filtered_gtFine_{dataset_settype}.json',
        data_prefix=dict(img=f'leftImg8bit/{dataset_settype}/'),
    )
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    # register_all_modules()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = argparse.ArgumentParser(description='Generate SAM patches')
    parser.add_argument('--dst', help='save path',
                        default='./segment_anything_main/output/debug')
    parser.add_argument('--ann', default=None)
    parser.add_argument("--dump_iters", default=5000)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument('--dataset_settype', default='train2017')
    parser.add_argument('--model_type', default='vit_h')
    args = parser.parse_args()

    ## distributed cuda
    local_rank = int(args.local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    sam_checkpoint = "/home/cpf/codes/SAM-CP/pretrained/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam = sam.to(device=device)

    setting_list = ['region_clip_coco_sam_orisize', 'region_clip_coco_sam_orisize_side64',
                    'region_clip_coco_sam_orisize_side48', 'region_clip_coco_sam_orisize_iou0.8',
                    'region_clip_coco_sam_orisize_stab0.85',
                    'region_clip_coco_sam_orisize_nms_0.8',
                    'region_clip_coco_sam_1333*800', 'region_clip_coco_sam_1333*800_side64',
                    'region_clip_coco_sam_orisize_side48_stab0.9_iou0.85',
                    'region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8',
                    'region_clip_coco_sam_1333*800_side64_stab0.9_iou0.8',
                    'region_clip_coco_sam_orisize_side48_stab0.9_iou0.8',
                    'region_clip_coco_sam_orisize_side64_stab0.9_iou0.8']
    setting = setting_list[9]
    if setting == 'region_clip_coco_sam_orisize':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 32, 0.88, 0.95, 0.7
    elif setting == 'region_clip_coco_sam_orisize_side64':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 64, 0.88, 0.95, 0.7
    elif setting == 'region_clip_coco_sam_orisize_side48':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 48, 0.88, 0.95, 0.7
    elif setting == 'region_clip_coco_sam_orisize_iou0.8':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 32, 0.8, 0.95, 0.7
    elif setting == 'region_clip_coco_sam_orisize_stab0.85':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 32, 0.88, 0.85, 0.7
    elif setting == 'region_clip_coco_sam_orisize_nms_0.8':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 32, 0.88, 0.95, 0.8
    elif setting == 'region_clip_coco_sam_1333*800':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = True, 32, 0.88, 0.95, 0.7
    elif setting == 'region_clip_coco_sam_1333*800_side64':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = True, 64, 0.88, 0.95, 0.7
    elif setting == 'region_clip_coco_sam_orisize_side48_stab0.9_iou0.85':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 48, 0.85, 0.9, 0.7
    elif setting == 'region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = True, 48, 0.8, 0.9, 0.7
    elif setting == 'region_clip_coco_sam_1333*800_side64_stab0.9_iou0.8':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = True, 64, 0.8, 0.9, 0.7
    elif setting == 'region_clip_coco_sam_orisize_side48_stab0.9_iou0.8':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 48, 0.8, 0.9, 0.7
    elif setting == 'region_clip_coco_sam_orisize_side64_stab0.9_iou0.8':
        with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = False, 64, 0.8, 0.9, 0.7

    #     setting = 'region_clip_coco_sam_orisize_side48_stab0.5_iou0.5_nms0.5'
    #     with_resize, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh = True, 48, 0.8, 0.9, 0.7
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side,
                                               points_per_batch=256,
                                               pred_iou_thresh=pred_iou_thresh,
                                               crop_n_layers=1,
                                               stability_score_thresh=stability_score_thresh,
                                               box_nms_thresh=box_nms_thresh).cuda()
    mask_generator = DistributedDataParallel(
        mask_generator.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # cfg['ann_file'] = args.ann if args.ann is not None else cfg['ann_file']
    data_loader, dataset = data_builder(cfg)

    if not os.path.isdir('output/'):
        os.mkdir('output/')
    output_dict = f'output/overlaps_cityscapes_{dataset_settype}_' + model_type + '_' + setting + '/'
    output_proposal_dict = f'output/proposals_cityscapes_{dataset_settype}_' + model_type + '_' + setting + '/'

    output_semantic_dict = f'output/overlaps_cityscapes_{dataset_settype}_' + model_type + '_' + setting + '/panoptic_overlaps/'

    semantic_dict = f'/cache/cityscapes/panoptic_{dataset_settype}/'
    gt_annotations_panoptic_file = f'/cache/cityscapes/annotations/panoptic_{dataset_settype}.json'
    # coco = COCOPanoptic(gt_annotations_panoptic_file)
    # cat_ids = coco.getCatIds(
    #     catNms=METAINFO['classes'])
    # cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    mode = ['bbox', 'segm']
    calculate_mode = ['iou', 'iof', 'iog']

    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)
    if not os.path.isdir(output_semantic_dict):
        os.mkdir(output_semantic_dict)
    if not os.path.isdir(output_proposal_dict):
        os.mkdir(output_proposal_dict)

    data_list = []
    det_result_list = []
    segm_result_list = []
    all_result_list = []
    img_ids = []
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # if i < 4998:
        #     continue
        image = data['inputs'][0].type(torch.uint8).permute(1, 2, 0).numpy()[:, :, ::-1]
        # data_samples = data['data_samples'][0]
        data_samples = data['data_samples'][0]
        img_id = data_samples.img_path.split('/')[-1].split('.')[0]
        f_name = '0' * (12 - len(str(img_id))) + str(img_id)
        output_proposal_file = output_proposal_dict + f_name + '.pkl'
        if os.path.isfile(output_proposal_file):
            print('aaa')
            continue
        masks = mask_generator(image)
        img_id = data_samples.img_path.split('/')[-1].split('.')[0]
        ## obtain segms
        segms = []
        for mask in masks:
            segms.append(mask['segmentation'])

        segms = np.array(segms, dtype=np.uint8)
        segms = BitmapMasks(segms, height=segms.shape[1], width=segms.shape[2])

        scale_factor = [1 / s for s in data_samples.scale_factor]
        new_scale = [math.ceil(x * y) for x, y in zip(scale_factor, segms.masks.shape[1:])]
        segms = segms.rescale(tuple(new_scale)).crop(np.array(
            [0, 0, data_samples.ori_shape[1], data_samples.ori_shape[0]]))
        mask_rle = [coco_encode_rle(rle) for rle in
                    mask_to_rle_pytorch(torch.tensor(segms.masks))]

        ## obtain bboxes
        pred_bboxes = box_xywh_to_xyxy(
            torch.cat([torch.tensor(m['bbox']).unsqueeze(0) for m in masks])).float()

        # bboxes = bbox_rescale(pred_bboxes, scale_factor)
        bboxes = pred_bboxes * pred_bboxes.new_tensor(scale_factor).repeat(2)
        from extra_tools.transform import xyxy_to_x1y1wh
        bboxes = xyxy_to_x1y1wh(bboxes)
        scores = torch.cat([torch.tensor(m['predicted_iou']).unsqueeze(0) for m in masks]).float()
        labels = torch.zeros(scores.shape).long()

        h, w = data_samples.ori_shape[0], data_samples.ori_shape[1]
        # shape=cv2.imread('/cache/coco/train2017/'+'0'*(12-len(str(results[0]['image_id'])))+str(results[0]['image_id'])+'.jpg').shape
        if h != mask_rle[0]['size'][0] or w != mask_rle[0]['size'][1]:
            for j in mask_rle:
                j['size'][0] = h
                j['size'][1] = w
            print((h, w), mask_rle[0]['size'])

        per_img = dict()
        per_img.update({'indexes': img_id, 'boxes': bboxes, 'segmentations': mask_rle, 'scores': scores})

        ### save the proposals
        f_name = '0' * (12 - len(str(img_id))) + str(img_id)
        output_proposal_file = output_proposal_dict + f_name + '.pkl'
        pickle.dump(per_img, open(output_proposal_file, 'wb'))

    # for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
    #     with torch.no_grad():
    #         data_samples = data['data_samples'][0]
    #         img_id = data_samples.img_path.split('/')[-1].split('.')[0]
    #         f_name = '0' * (12 - len(str(img_id))) + str(img_id)
    #         output_proposal_file = output_proposal_dict + f_name + '.pkl'
    #         per_img = pickle.load(open(output_proposal_file, 'rb'))
    #         gt_semantic_seg=preprocess_gt([data_samples.gt_instances], [data_samples.gt_sem_seg])[0]
    #         gt_masks, gt_bboxes=data_samples.gt_instances['masks'],data_samples.gt_instances['bboxes']
    #         gt_img_info=data_samples.metainfo
    #
    #         calcualte_overlap(per_img, [gt_img_info, gt_semantic_seg], mode=['segm', ],
    #                           calculate_mode=['iou', 'iof', 'iog'], output_dict=output_semantic_dict + 'semantic/')
    #
    #         calcualte_overlap(per_img, [gt_img_info, data_samples.gt_instances], output_dict=output_semantic_dict + 'instance/')

# python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 \
# --master_addr='127.0.0.1' --master_port='29500' train.py
# PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr='127.0.0.1' --master_port='29500' segment_anything_main/test/seg_generation.py
