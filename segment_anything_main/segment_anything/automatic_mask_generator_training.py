# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple
from mmdet.registry import MODELS
from .modeling import Sam
from .predictor_training import SamPredictor_training
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from segment_anything_main.segment_anything import sam_model_registry
from mmengine.structures import InstanceData
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models.utils import empty_instances, multi_apply
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.models.layers import multiclass_nms


class SamAutomaticMaskGenerator_training:
    def __init__(
            self,
            model: Sam,
            points_per_side: Optional[int] = 32,
            points_per_batch: int = 64,
            pred_iou_thresh: float = 0.88,
            stability_score_thresh: float = 0.95,
            stability_score_offset: float = 1.0,
            box_nms_thresh: float = 0.7,
            crop_n_layers: int = 0,
            crop_nms_thresh: float = 0.7,
            crop_overlap_ratio: float = 512 / 1500,
            crop_n_points_downscale_factor: int = 1,
            point_grids: Optional[List[np.ndarray]] = None,
            min_mask_region_area: int = 0,
            output_mode: str = "binary_mask",
            train_cfg=None,
            test_cfg=None,
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
                point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
            self.point_grids = torch.tensor(self.point_grids).cuda()
        elif point_grids is not None:
            self.point_grids = point_grids
            self.point_grids = torch.tensor(self.point_grids).cuda()
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor_training(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.sam_model = model
        self.loss_cls = MODELS.build(model.loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_assigner = TASK_UTILS.build(train_cfg.assigner)
        self.patch_sampler = TASK_UTILS.build(train_cfg.sampler, default_args=dict(context=self))

    def set_image(self, batch_inputs):
        # Pad
        h, w = batch_inputs.shape[-2:]
        padh = self.sam_model.image_encoder.img_size - h
        padw = self.sam_model.image_encoder.img_size - w
        x = F.pad(batch_inputs, (0, padw, 0, padh))
        return x

    def predict_torch(
            self,
            point_coords: Optional[torch.Tensor],
            point_labels: Optional[torch.Tensor],
            boxes: Optional[torch.Tensor] = None,
            mask_input: Optional[torch.Tensor] = None,
            orig_size=None,
            input_size=None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions, cls_predictions = self.sam_model.mask_decoder(
            image_embeddings=self.features.detach(),
            image_pe=self.sam_model.prompt_encoder.get_dense_pe().detach(),
            sparse_prompt_embeddings=sparse_embeddings.detach(),
            dense_prompt_embeddings=dense_embeddings.detach(),
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.sam_model.postprocess_masks(low_res_masks, input_size, orig_size)

        if not return_logits:
            masks = masks > self.sam_model.mask_threshold

        return masks, iou_predictions, low_res_masks, cls_predictions

    def spot_sampling(self, batch_data_samples, num_sample=10):
        spot_prompt_list = []
        negative_list = []
        point_sampling_list = []
        num_gt = [len(data_samples.gt_instances.labels) for data_samples in batch_data_samples]
        for j, data_samples in enumerate(batch_data_samples):
            gt_instances = data_samples.gt_instances
            gt_masks = gt_instances.masks
            spot_prompt = []
            for i in range(len(gt_masks)):
                spot_pts = np.transpose(np.nonzero(gt_masks.masks[i]))[:, ::-1]
                index = np.random.randint(0, len(spot_pts), num_sample)
                chosen_pts = spot_pts[index]
                spot_prompt.append(chosen_pts)
            spot_prompt = np.stack(spot_prompt)
            spot_prompt_list.append(spot_prompt)

            max_num_gt = max(num_gt)
            neg_pts = np.transpose(np.nonzero(1 - gt_masks.masks.max(0)))
            neg_index = np.random.randint(0, len(neg_pts), num_sample * (2 * max_num_gt - num_gt[j]))
            negative_list.append(neg_pts[neg_index])

            point_sampling_list.append(np.concatenate([spot_pts.reshape(-1, 2), neg_pts[neg_index]]))
        return spot_prompt_list, negative_list, point_sampling_list

    def loss(self, batch_inputs, batch_data_samples):

        spot_prompt_list, negative_list, point_sampling_list = self.spot_sampling(batch_data_samples,
                                                                                  num_sample=self.train_cfg.num_spot_sample)
        point_sampling_list = [torch.tensor(i, device=batch_inputs.device) for i in point_sampling_list]
        # Get points for this crop
        orig_size = batch_inputs.shape[-2:]
        # points_scale = torch.tensor(orig_size)[None, [1, 0]].to(batch_inputs.device)
        # points_for_image = self.point_grids[0] * points_scale
        batch_inputs = self.set_image(batch_inputs)
        self.features = self.sam_model.image_encoder(batch_inputs)
        # self.features = torch.cat([i.img_features for i in batch_data_samples])
        point_labels = [torch.ones(points_for_image.shape[0], dtype=torch.int, device=batch_inputs.device) for
                        points_for_image in point_sampling_list]
        all_for_img = (points_for_image[:, None], point_labels[:, None], None, None)
        cond = [x is not None for x in all_for_img]
        all_for_img = [x for x in all_for_img if x is not None]

        masks, iou_predictions, low_res_masks, cls_predictions, pred_bboxs = [], [], [], [], []
        for batches in batch_iterator(self.points_per_batch, *all_for_img):
            points_, labels_, boxes_, mask_input_ = [batches.pop(0) if x else None for x in cond]
            masks_, iou_predictions_, low_res_masks_, cls_predictions_ = self.predict_torch(
                point_coords=points_,
                point_labels=labels_,
                boxes=boxes_,
                mask_input=mask_input_,
                orig_size=orig_size,
                input_size=orig_size, )
            # masks.append(masks_)

            iou_predictions.append(iou_predictions_)
            # low_res_masks.append(low_res_masks_)
            cls_predictions.append(cls_predictions_)
            pred_bboxs.append(batched_mask_to_box(masks_))
            del masks_
        # masks = torch.cat(masks, dim=0)
        bboxs = torch.cat(pred_bboxs, dim=0)
        iou_predictions = torch.cat(iou_predictions, dim=0)
        # low_res_masks = torch.cat(low_res_masks, dim=0)
        cls_predictions = torch.cat(cls_predictions, dim=0)

        # reshape to batch
        # masks = masks.reshape(-1, points_for_image.shape[0], *masks.shape[1:])
        iou_predictions = iou_predictions.reshape(-1, points_for_image.shape[0], *iou_predictions.shape[1:])
        # low_res_masks = low_res_masks.reshape(-1, points_for_image.shape[0], *low_res_masks.shape[1:])
        cls_predictions = cls_predictions.reshape(-1, points_for_image.shape[0], *cls_predictions.shape[1:])
        bboxs = bboxs.reshape(-1, points_for_image.shape[0], *bboxs.shape[1:])

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            results = InstanceData()
            results.points = points_for_image.repeat_interleave(bboxs[i].shape[1], dim=0)
            results.bboxes = bboxs[i].flatten(0, 1)
            results.priors = bboxs[i].flatten(0, 1)
            # results.masks = masks[i].flatten(0, 1)
            results.cls_scores = cls_predictions[i].flatten(0, 1)
            results.iou_scores = iou_predictions[i].flatten(0, 1)

            assign_result = self.patch_assigner.assign(
                results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.patch_sampler.sample(
                assign_result,
                results,
                batch_gt_instances[i])
            sampling_results.append(sampling_result)

        losses = dict()

        labels, label_weights = multi_apply(self.get_target, sampling_results)
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        cls_score = torch.cat([cls_predictions[i].flatten(0, 1) for i in range(len(sampling_results))])
        # losses.update(bbox_results['loss_bbox'])
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        losses['loss_cls'] = loss_cls

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale):

        # Get points for this crop
        orig_size = batch_inputs.shape[-2:]
        points_scale = torch.tensor(orig_size)[None, [1, 0]].to(batch_inputs.device)
        points_for_image = self.point_grids[0] * points_scale
        batch_inputs = self.set_image(batch_inputs)
        self.features = self.sam_model.image_encoder(batch_inputs)
        # self.features =batch_data_samples[0].img_features
        # dir = 'work_dirs/sam_feature/coco/train2017/'
        # torch.save(self.features, dir + batch_data_samples[0].img_path.split('/')[-1].split('.')[0] + 'pth')
        # torch.save(self.features,dir+batch_data_samples[0].img_path.split('/')[-1].split('.')[0])
        point_labels = torch.ones(points_for_image.shape[0], dtype=torch.int, device=points_for_image.device)
        all_for_img = (points_for_image[:, None], point_labels[:, None], None, None)
        cond = [x is not None for x in all_for_img]
        all_for_img = [x for x in all_for_img if x is not None]

        masks, iou_predictions, low_res_masks, cls_predictions, pred_bboxs = [], [], [], [], []
        for batches in batch_iterator(self.points_per_batch, *all_for_img):
            points_, labels_, boxes_, mask_input_ = [batches.pop(0) if x else None for x in cond]
            masks_, iou_predictions_, low_res_masks_, cls_predictions_ = self.predict_torch(
                point_coords=points_,
                point_labels=labels_,
                boxes=boxes_,
                mask_input=mask_input_,
                orig_size=orig_size,
                input_size=orig_size, )
            # masks.append(masks_)

            iou_predictions.append(iou_predictions_)
            # low_res_masks.append(low_res_masks_)
            cls_predictions.append(cls_predictions_)
            pred_bboxs.append(batched_mask_to_box(masks_))
            del masks_
        # masks = torch.cat(masks, dim=0)
        bboxs = torch.cat(pred_bboxs, dim=0)
        iou_predictions = torch.cat(iou_predictions, dim=0)
        # low_res_masks = torch.cat(low_res_masks, dim=0)
        cls_predictions = torch.cat(cls_predictions, dim=0)

        # reshape to batch
        # masks = masks.reshape(-1, points_for_image.shape[0], *masks.shape[1:])
        iou_predictions = iou_predictions.reshape(-1, points_for_image.shape[0], *iou_predictions.shape[1:])
        # low_res_masks = low_res_masks.reshape(-1, points_for_image.shape[0], *low_res_masks.shape[1:])
        cls_predictions = cls_predictions.reshape(-1, points_for_image.shape[0], *cls_predictions.shape[1:])
        bboxs = bboxs.reshape(-1, points_for_image.shape[0], *bboxs.shape[1:])

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        result_list = self.predict_bbox(batch_img_metas, cls_predictions.sigmoid(), bbox_predictions=bboxs,
                                        rescale=rescale)

        return result_list

    def predict_bbox(self, batch_img_metas, cls_predictions, bbox_predictions, rescale):
        num_imgs = len(batch_img_metas)

        result_list_det = []
        for img_id in range(num_imgs):
            results = InstanceData()
            cls_prediction = cls_predictions[img_id].flatten(0, 1)
            bboxes = bbox_predictions[img_id].flatten(0, 1).float()
            img_meta = batch_img_metas[img_id]
            # mask_prediction = masks[i]
            # img_shape = img_meta['img_shape']
            # if img_shape is not None and bboxes.size(-1) == 4:
            #     bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
            #     bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
            if rescale and bboxes.size(0) > 0:
                assert img_meta.get('scale_factor') is not None
                scale_factor = [1 / s for s in img_meta['scale_factor']]
                bboxes = scale_boxes(bboxes, scale_factor)
            sam_cfg = dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)
            # from mmdet.utils import OptConfigType
            # sam_cfg =OptConfigType(sam_cfg)
            if sam_cfg is not None:
                box_dim = bboxes.size(-1)
                det_bboxes, det_labels, idx = multiclass_nms(
                    bboxes,
                    cls_prediction,
                    sam_cfg['score_thr'],
                    sam_cfg['nms'],
                    sam_cfg['max_per_img'],
                    return_inds=True,
                    box_dim=box_dim)
                results.bboxes = det_bboxes[:, :-1]
                results.scores = det_bboxes[:, -1]
                results.labels = det_labels

            result_list_det.append(results)
        return result_list_det

    def get_target(self, sampling_results):
        pos_gt_labels = sampling_results.pos_gt_labels
        num_pos = sampling_results.pos_priors.size(0)
        num_neg = sampling_results.neg_priors.size(0)
        num_samples = num_pos + num_neg
        pos_inds = sampling_results.pos_inds
        neg_inds = sampling_results.neg_inds
        labels = pos_gt_labels.new_full((num_samples,),
                                        self.sam_model.mask_decoder.num_classes,
                                        dtype=torch.long)
        label_weights = pos_gt_labels.new_zeros(num_samples)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if self.train_cfg.pos_weight <= 0 else self.train_cfg.pos_weight
            label_weights[pos_inds] = pos_weight
        if num_neg > 0:
            label_weights[neg_inds] = 1.0
        return labels, label_weights

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[2:]

        # Iterate over image crops

        crop_data = self._process_crop(image, orig_size)

        return crop_data

    def _process_crop(
            self,
            image: np.ndarray,
            orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings

        self.predictor.set_image(image)

        # Get points for this crop
        points_scale = np.array(orig_size)[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        data_baches = []
        # Generate masks for this crop in batches
        data = [MaskData() for i in range(len(image))]
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, orig_size)
            for i, bd in enumerate(batch_data):
                data[i].cat(bd)
            del batch_data
        self.predictor.reset_image()
        orig_w, orig_h = orig_size
        # Remove duplicates within this crop.
        for i in range(len(image)):
            keep_by_nms = batched_nms(
                data[i]["boxes"].float(),
                data[i]["iou_preds"],
                torch.zeros_like(data[i]["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data[i].filter(keep_by_nms)

            # Return to the original image frame
            data[i]["boxes"] = uncrop_boxes_xyxy(data[i]["boxes"], [0, 0, orig_w, orig_h])
            data[i]["points"] = uncrop_points(data[i]["points"], [0, 0, orig_w, orig_h])
            # data[i]["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data[i]["rles"]))])

        return data

    def _process_batch(
            self,
            points: np.ndarray,
            orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, orig_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        masks = masks.reshape(-1, points.shape[0], *masks.shape[1:])
        iou_preds = iou_preds.reshape(-1, points.shape[0], *iou_preds.shape[1:])
        # Serialize predictions and store in MaskData

        data_list = [MaskData(
            masks=m.flatten(0, 1),
            iou_preds=i.flatten(0, 1),
            points=torch.as_tensor(points.repeat(m.shape[1], axis=0)),
        ) for m, i in zip(masks, iou_preds)]
        del masks

        for i in range(len(data_list)):
            data = data_list[i]
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

            # Threshold masks and calculate boxes
            data["masks"] = data["masks"] > self.predictor.model.mask_threshold
            data["boxes"] = batched_mask_to_box(data["masks"])

            # Filter boxes that touch crop boundaries
            keep_mask = ~is_box_near_crop_edge(data["boxes"], [0, 0, orig_w, orig_h], [0, 0, orig_w, orig_h])
            if not torch.all(keep_mask):
                data.filter(keep_mask)

            # Compress to RLE
            data["masks"] = uncrop_masks(data["masks"], [0, 0, orig_w, orig_h], orig_h, orig_w)
            data["rles"] = mask_to_rle_pytorch(data["masks"])
            del data["masks"]

        return data_list

    @staticmethod
    def postprocess_small_regions(
            mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
