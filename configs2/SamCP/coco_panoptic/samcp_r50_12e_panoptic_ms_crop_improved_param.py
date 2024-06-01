_base_ = [
    '../../../configs/_base_/datasets/coco_detection.py', '../../../configs/_base_/default_runtime.py',
]
model = dict(
    type='SamCP',
    num_queries=300,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=False,
    use_image_encoder=False,
    use_patch_query=True,
    use_denosing=True,
    use_patch_encoder=True,
    use_shared_mlp=True,
    use_mask_roi=True,
    num_feature_levels=5,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    # neck=dict(
    #     type='ChannelMapper',
    #     in_channels=[512, 1024, 2048],
    #     kernel_size=1,
    #     out_channels=256,
    #     act_cfg=None,
    #     norm_cfg=dict(type='GN', num_groups=32),
    #     num_outs=5),  ## 8,16,32,64
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=5,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    encoder_patch=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        use_dab_querypos=True,
        pos_temperature=10000,
        use_affinity_refine=True,
        share_affinity=False,
        layer_cfg=dict(
            use_patch_cross=False,
            query_cross_patch=True,
            use_query_pos_affinity=True,
            use_deformable=True,
            cross_attn_cfg=dict(embed_dims=256, num_heads=8,
                                dropout=0.0),  # 0.1 for DeformDETR
            # cross_patch_attn_cfg=dict(embed_dims=256, num_heads=8,
            #                     dropout=0.0),  # 0.1 for DeformDETR
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=-0.5,  # -0.5 for DeformDETR
        temperature=10000),  # 10000 for DeformDETR
    bbox_head=dict(
        type='SamCPHead',
        num_classes=80,
        use_patch_cls=False,
        shared_cls=True,
        use_regression=False,
        sync_cls_avg_factor=True,
        use_o2o_cls=True,
        instace_cfg=dict(category_embeddings='pretrained/clip/clip_categories_embedding/coco_80_with_bg_vit-b.pth',
                         num_thing_classes=80,
                         ),
        semantic_cfg=dict(
            category_embeddings='pretrained/clip/clip_categories_embedding/coco_semantic_53_vit-b.pth',
            semantic_offline_overlap='pretrained/sam_proposals/coco_train2017/overlaps_vit_h_region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8/panoptic_overlaps/semantic/',
            num_stuff_classes=53,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),  # 2.0 in DeformDETR
        loss_affinity=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=300)),  # TODO: half num_dn_queries

    # training and testing settings
    # train_cfg=dict(
    #     assigner=dict(
    #         type='HungarianAssigner',
    #         match_costs=[
    #             dict(type='FocalLossCost', weight=2.0),
    #             dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
    #             dict(type='IoUCost', iou_mode='giou', weight=2.0)
    #         ])),
    # test_cfg=dict(max_per_img=300))  # 100 for DeformDETR
    #
    # roi_head=dict(
    #     type='SamInstAttnRoIHead',
    #     bbox_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    #         out_channels=256,
    #         featmap_strides=[4, 8, 16, 32]),
    #     bbox_head=dict(
    #         type='ClipBBoxHead',
    #         # clip_type='ViT-B/32',
    #         clip_type='/cache/pretrained/clip/ViT-B-32.pt',
    #         extract_feat_mode='RoI_Feat_Attn',
    #         num_shared_fcs=2,
    #         in_channels=256,
    #         shared_fc_channels=1024,
    #         out_channels=512,
    #         roi_feat_size=7,
    #         classification_mode='text',
    #         category_embeddings='/cache/pretrained/clip/clip_categories_embedding/coco_80_with_bg.pth',
    #         num_classes=80,
    #         bbox_coder=dict(
    #             type='DeltaXYWHBBoxCoder',
    #             target_means=[0., 0., 0., 0.],
    #             target_stds=[0.1, 0.1, 0.2, 0.2]),
    #         reg_class_agnostic=True,
    #         # loss_cls=dict(
    #         # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #         loss_cls=dict(
    #             type='FocalLoss',
    #             use_sigmoid=True,
    #             gamma=2.0,
    #             alpha=0.25,
    #             loss_weight=1.0),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        refine_assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ]),
        assigner=dict(
            type='PartAssignerPlus',
            offline_assign=dict(
                offline_overlap='pretrained/sam_proposals/coco_train2017/overlaps_vit_h_region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8/panoptic_overlaps/instance/',
                proposal_mode=['bbox', 'segm'],
                calculate_mode='iof', ),
            pos_iou_thr=0.8,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=True,
            ignore_iof_thr=-1),
        # sampler=dict(type='PseudoSampler', add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False,
        with_score_norm=True,
        query_assigner_cfg=dict(
            ass_type='one2one',
            weighted=True,
            ass_cost=dict(cls_cost=4.0, affinity_cost=2.0, bbox_cost=2.0, iou_cost=2.0, dice_cost=2.0)
        ), ),
    test_cfg=dict(
        max_per_img=100,
        iou_thr=0.8,
        filter_low_score=True,
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
    ))

# dataset settings
dataset_type = 'CocoPanopticPPSDataset'
data_root = 'data/coco/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=(384, 600),
    #     allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/panoptic_train2017.json',
        # proposal_file='/home/pfchen/disk1/cpf/SamInst-main/output/proposals/coco/region_clip_coco_sam_1333*800_side48_stab0.pkl',
        data_prefix=dict(
            img='images/', seg='panoptic_train2017/'),
        proposal_dict='pretrained/sam_proposals/coco_train2017/proposals_vit_h_region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8/',
        # proposal_dict='output/crop_proposals_coco_val2017_vit_h_region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8/',

        proposal_cfg=dict(proposals_segm_mode='RoI', roi_segm_size=7),  ## Ori_Img
        # proposal_cfg=dict(proposals_segm_mode='OriImage_'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/panoptic_val2017.json',
        # proposal_file='/home/pfchen/disk1/cpf/SamInst-main/output/proposals/coco_val/7_region_clip_coco_sam_1333*800_side48_stab0.pkl',
        # data_prefix=dict(img='val2017/'),
        data_prefix=dict(img='images/', seg='panoptic_val2017/'),
        proposal_dict='pretrained/sam_proposals/coco_val2017/proposals_vit_h_region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8/',
        # proposal_dict='output/crop_proposals_coco_val2017_vit_h_region_clip_coco_sam_1333*800_side48_stab0.9_iou0.8/',

        # proposal_cfg=dict(proposals_segm_mode='RoI', roi_segm_size=7),  ## Ori_Img
        proposal_cfg=dict(proposals_segm_mode='OriImage'),
        # proposal_file='proposals/rpn_r50_fpn_1x_train2017.pkl',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = [dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args),
    dict(
        type='CocoPanopticMetric',
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        seg_prefix=data_root + 'panoptic_val2017/',
        backend_args={{_base_.backend_args}}),
    dict(type='SemSegMetric', iou_metrics=['mIoU'])
]
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        })
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

find_unused_parameters = True