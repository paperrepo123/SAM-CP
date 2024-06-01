# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union, Any
from pycocotools import mask as maskUtils
from mmdet.structures.mask import BitmapMasks
from mmdet.structures.bbox import HorizontalBoxes
import numpy as np
from mmengine.fileio import load
from mmdet.registry import DATASETS
from .api_wrappers import COCOPanoptic
from .coco import CocoDataset
import mmcv

@DATASETS.register_module()
class CocoPanopticPPSDataset(CocoDataset):
    """Coco dataset for Panoptic segmentation.

    The annotation format is shown as follows. The `ann` field is optional
    for testing.

    .. code-block:: none

        [
            {
                'filename': f'{image_id:012}.png',
                'image_id':9
                'segments_info':
                [
                    {
                        'id': 8345037, (segment_id in panoptic png,
                                        convert from rgb)
                        'category_id': 51,
                        'iscrowd': 0,
                        'bbox': (x1, y1, w, h),
                        'area': 24315
                    },
                    ...
                ]
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            ``dict(img=None, ann=None, seg=None)``. The prefix ``seg`` which is
            for panoptic segmentation map must be not None.
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """

    METAINFO = {
        'classes':
            ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
             'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
             'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
             'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
             'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
             'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
             'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
             'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
             'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
             'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
             'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
             'wall-wood', 'water-other', 'window-blind', 'window-other',
             'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
             'cabinet-merged', 'table-merged', 'floor-other-merged',
             'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
             'paper-merged', 'food-other-merged', 'building-other-merged',
             'rock-merged', 'wall-other-merged', 'rug-merged'),
        'thing_classes':
            ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
             'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
             'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
             'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
             'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
             'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        'stuff_classes':
            ('banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
             'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
             'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
             'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
             'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
             'wall-wood', 'water-other', 'window-blind', 'window-other',
             'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
             'cabinet-merged', 'table-merged', 'floor-other-merged',
             'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
             'paper-merged', 'food-other-merged', 'building-other-merged',
             'rock-merged', 'wall-other-merged', 'rug-merged'),
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
             (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
             (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
             (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
             (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
             (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
             (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
             (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
             (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
             (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
             (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
             (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
             (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
             (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
             (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
             (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
             (246, 0, 122), (191, 162, 208), (255, 255, 128), (147, 211, 203),
             (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100),
             (92, 136, 89), (218, 88, 184), (241, 129, 0), (217, 17, 255),
             (124, 74, 181), (70, 70, 70), (255, 228, 255), (154, 208, 0),
             (193, 0, 92), (76, 91, 113), (255, 180, 195), (106, 154, 176),
             (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
             (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
             (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
             (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
             (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
             (146, 139, 141), (70, 130, 180), (134, 199, 156), (209, 226, 140),
             (96, 36, 108), (96, 96, 96), (64, 170, 64), (152, 251, 152),
             (208, 229, 228), (206, 186, 171), (152, 161, 64), (116, 112, 0),
             (0, 114, 143), (102, 102, 156), (250, 141, 255)]
    }
    COCOAPI = COCOPanoptic
    # ann_id is not unique in coco panoptic dataset.
    ANN_ID_UNIQUE = False

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=None, ann=None, seg=None),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 proposal_file: Optional[str] = None,
                 proposal_dict: Optional[str] = None,
                 proposal_cfg: dict = None,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 backend_args: dict = None,
                 **kwargs) -> None:
        self.proposal_file = proposal_file
        self.proposal_dict = proposal_dict
        self.proposal_cfg = proposal_cfg
        from timer import Timer
        self.e = Timer()
        self.f = Timer()
        self.t = Timer()
        self.g = Timer()
        self.h=Timer()
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            backend_args=backend_args,
            **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        # filter out unmatched annotations which have
        # same segment_id but belong to other image
        ann_info = [
            ann for ann in ann_info if ann['image_id'] == img_info['img_id']
        ]
        data_info = {}

        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].replace('jpg', 'png'))
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['thing_classes']
            data_info['stuff_text'] = self.metainfo['stuff_classes']
            data_info['custom_entities'] = True  # no important

        instances = []
        segments_info = []
        for ann in ann_info:
            instance = {}
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                instance['bbox'] = bbox
                instance['bbox_label'] = contiguous_cat_id
                if not is_crowd:
                    instance['ignore_flag'] = 0
                else:
                    instance['ignore_flag'] = 1
                    is_thing = False

            segment_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            segments_info.append(segment_info)
            if len(instance) > 0 and is_thing:
                instances.append(instance)
        data_info['instances'] = instances
        data_info['segments_info'] = segments_info
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter images too small or without ground truth.

        Returns:
            List[dict]: ``self.data_list`` after filtering.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = set()
        # check whether images have legal thing annotations.
        for data_info in self.data_list:
            for segment_info in data_info['segments_info']:
                if not segment_info['is_thing']:
                    continue
                ids_with_ann.add(data_info['img_id'])

        valid_data_list = []
        for data_info in self.data_list:
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(width, height) >= min_size:
                valid_data_list.append(data_info)

        return valid_data_list

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)

        if self.proposal_dict is not None:
            f_name = '0' * (12 - len(str(data_info['img_id']))) + str(data_info['img_id'])
            proposals = load(self.proposal_dict + f_name + '.pkl')
            proposals_boxes = x1y1wh_to_xyxy(proposals['boxes']).astype(np.float32)
            data_info['proposals_boxes'] = HorizontalBoxes(proposals_boxes)

            h, w = data_info['height'], data_info['width']

            # self.e.tic('pps')

            if self.proposal_cfg['proposals_segm_mode'] == 'OriImage':
                data_info['proposals_segms'] = BitmapMasks(
                    maskUtils.decode(proposals['segmentations']).transpose(2, 0, 1),
                    h,
                    w)
                data_info['proposals_segm_mode'] = 'OriImage'
                size = self.proposal_cfg['roi_segm_size'] if 'roi_segm_size' in self.proposal_cfg else 7
                num_proposals = len(proposals_boxes)
                data_info['roi_segms'] = data_info['proposals_segms'].crop_and_resize(proposals_boxes,
                                                                                      (size, size),
                                                                                      np.arange(num_proposals))
            elif self.proposal_cfg['proposals_segm_mode'] == 'AugImage':
                data_info['proposals_segms'] = BitmapMasks(
                    maskUtils.decode(proposals['segmentations']).transpose(2, 0, 1),
                    h,
                    w)
                data_info['proposals_segm_mode'] = 'AugImage'
                size = self.proposal_cfg['roi_segm_size'] if 'roi_segm_size' in self.proposal_cfg else 7
                num_proposals = len(proposals_boxes)
                data_info['roi_segms'] = data_info['proposals_segms'].crop_and_resize(proposals_boxes,
                                                                                      (size, size),
                                                                                      np.arange(num_proposals))

            elif self.proposal_cfg['proposals_segm_mode'] == 'RoI':
                data_info['proposals_segm_mode'] = 'RoI'
                size = self.proposal_cfg['roi_segm_size']
                num_proposals = len(proposals_boxes)
                # self.t.tic('mde')
                mk = maskUtils.decode(proposals['segmentations'])
                # self.t.toc('mde')
                mk = mk.transpose(2, 0, 1)
                # self.g.tic('bits')
                xxxx = BitmapMasks(mk, h, w)
                # self.g.toc('bits')

                # self.f.tic('bitsss')
                data_info['roi_segms'] = xxxx.crop_and_resize(proposals_boxes, (size, size), np.arange(num_proposals))
                # data_info['check_parts']=
                # self.f.toc('bitsss')
            # self.e.toc('pps')
            # self.h.tic('ppline')
            dd = self.pipeline(data_info)
            # self.h.toc('ppline')
        return dd


def x1y1wh_to_xyxy(bbox):
    return np.stack([bbox[:, 0], bbox[:, 1], bbox[:, 2] + bbox[:, 0], bbox[:, 3] + bbox[:, 1]], axis=1)

# from timer import Timer
# e=Timer()
# e.tic('sss')
# proposals=load('/home/cpf/codes/SAM-CP/pretrained/sam_proposals/coco_train2017/proposals_vit_h_region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8/000000000201.pkl')
# proposals_boxes = x1y1wh_to_xyxy(proposals['boxes']).astype(np.float32)
# mk = maskUtils.decode(proposals['segmentations'])
# w,h=proposals['segmentations'][0]['size']
#
# xxxx = BitmapMasks(mk, h, w)
# num_proposals = len(proposals_boxes)
# xxxx.crop_and_resize(proposals_boxes, (size, size), np.arange(num_proposals))
# e.toc('sss')
#
