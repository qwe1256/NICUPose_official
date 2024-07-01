dataset_type = 'NICUPose.CocoPoseDataset'
data_root = 'C:/Users/Hugo/Study/Infant Dataset/NICU Image/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', to_float32=True),
    dict(
        type='NICUPose.LoadAnnotations',
        with_bbox=True,
        with_keypoint=True,
        with_area=True),
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='NICUPose.KeypointRandomAffine',
        max_rotate_degree=30.0,
        max_translate_ratio=0.0,
        scaling_ratio_range=(1.0, 1.0),
        max_shear_degree=0.0,
        border_val=[103.53, 116.28, 123.675]),
    dict(type='NICUPose.RandomFlip', flip_ratio=0.5),
    dict(
        type='mmdet.AutoAugment',
        policies=[[{
            'type': 'NICUPose.Resize',
            'img_scale': [(400, 1400), (1400, 1400)],
            'multiscale_mode': 'range',
            'keep_ratio': True
        }],
                  [{
                      'type': 'NICUPose.Resize',
                      'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'NICUPose.RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (384, 600),
                      'allow_negative_crop': True
                  }, {
                      'type': 'NICUPose.Resize',
                      'img_scale': [(400, 1400), (1400, 1400)],
                      'multiscale_mode': 'range',
                      'override': True,
                      'keep_ratio': True
                  }]]),
    dict(
        type='mmdet.Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='mmdet.Pad', size_divisor=1),
    dict(
        type='NICUPose.DefaultFormatBundle',
        extra_keys=['gt_keypoints', 'gt_areas']),
    dict(
        type='mmdet.Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'])
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(
                type='mmdet.Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img'])
        ])
]
demo_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(
                type='mmdet.Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='NICUPose.CocoPoseDataset',
        ann_file=
        'C:/Users/Hugo/Study/Infant Dataset/NICU Image/annotations/person_keypoints_train2017.json',
        img_prefix=
        'C:/Users/Hugo/Study/Infant Dataset/NICU Image/images/train2017/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', to_float32=True),
            dict(
                type='NICUPose.LoadAnnotations',
                with_bbox=True,
                with_keypoint=True,
                with_area=True),
            dict(
                type='mmdet.PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='NICUPose.KeypointRandomAffine',
                max_rotate_degree=30.0,
                max_translate_ratio=0.0,
                scaling_ratio_range=(1.0, 1.0),
                max_shear_degree=0.0,
                border_val=[103.53, 116.28, 123.675]),
            dict(type='NICUPose.RandomFlip', flip_ratio=0.5),
            dict(
                type='mmdet.AutoAugment',
                policies=[[{
                    'type': 'NICUPose.Resize',
                    'img_scale': [(400, 1400), (1400, 1400)],
                    'multiscale_mode': 'range',
                    'keep_ratio': True
                }],
                          [{
                              'type': 'NICUPose.Resize',
                              'img_scale': [(400, 4200), (500, 4200),
                                            (600, 4200)],
                              'multiscale_mode': 'value',
                              'keep_ratio': True
                          }, {
                              'type': 'NICUPose.RandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (384, 600),
                              'allow_negative_crop': True
                          }, {
                              'type': 'NICUPose.Resize',
                              'img_scale': [(400, 1400), (1400, 1400)],
                              'multiscale_mode': 'range',
                              'override': True,
                              'keep_ratio': True
                          }]]),
            dict(
                type='mmdet.Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(
                type='NICUPose.DefaultFormatBundle',
                extra_keys=['gt_keypoints', 'gt_areas']),
            dict(
                type='mmdet.Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'
                ])
        ]),
    val=dict(
        type='NICUPose.CocoPoseDataset',
        ann_file=
        'C:/Users/Hugo/Study/Infant Dataset/NICU Image/annotations/person_keypoints_val2017.json',
        img_prefix=
        'C:/Users/Hugo/Study/Infant Dataset/NICU Image/images/val2017/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                type='mmdet.MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='mmdet.Resize', keep_ratio=True),
                    dict(type='mmdet.RandomFlip'),
                    dict(
                        type='mmdet.Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='mmdet.Pad', size_divisor=1),
                    dict(type='mmdet.ImageToTensor', keys=['img']),
                    dict(type='mmdet.Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='NICUPose.CocoPoseDataset',
        ann_file=
        'C:/Users/Hugo/Study/Infant Dataset/NICU Image/annotations/person_keypoints_val2017.json',
        img_prefix=
        'C:/Users/Hugo/Study/Infant Dataset/NICU Image/images/val2017/',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                type='mmdet.MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='mmdet.Resize', keep_ratio=True),
                    dict(type='mmdet.RandomFlip'),
                    dict(
                        type='mmdet.Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='mmdet.Pad', size_divisor=1),
                    dict(type='mmdet.ImageToTensor', keys=['img']),
                    dict(type='mmdet.Collect', keys=['img'])
                ])
        ]),
    demo=dict(
        type='NICUPose.CocoPoseDataset',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                type='mmdet.MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='mmdet.Resize', keep_ratio=True),
                    dict(type='mmdet.RandomFlip'),
                    dict(
                        type='mmdet.Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='mmdet.Pad', size_divisor=1),
                    dict(type='mmdet.ImageToTensor', keys=['img']),
                    dict(type='mmdet.Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='keypoints')
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'
model = dict(
    type='NICUPose.Detector',
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=192,
        convert_weights=True,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'
        )),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='NICUPose.Head',
        num_query=300,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        transformer=dict(
            type='NICUPose.Transformer',
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='NICUPose.TransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='NICUPose.MultiScaleDeformablePoseAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            hm_encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_kpt=dict(type='mmdet.L1Loss', loss_weight=70.0),
        loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
        loss_oks=dict(type='NICUPose.OKSLoss', loss_weight=2.0),
        loss_hm=dict(type='NICUPose.CenterFocalLoss', loss_weight=4.0),
        loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=80.0),
        loss_oks_refine=dict(type='NICUPose.OKSLoss', loss_weight=3.0)),
    train_cfg=dict(
        assigner=dict(
            type='NICUPose.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            kpt_cost=dict(type='NICUPose.KptL1Cost', weight=70.0),
            oks_cost=dict(type='NICUPose.OksCost', weight=7.0))),
    test_cfg=dict(max_per_img=100))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[27], gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=300)
work_dir = './work_dirs\swin-l-p4-w7-224-22kto1k_16x1_100e_coco'
auto_resume = False
gpu_ids = [0]
