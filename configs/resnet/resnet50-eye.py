# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth',
            prefix='backbone.')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

# data settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2,  # 这里是后添加的：分类数目修改为2，猫和狗
    # RGB format normalization parameters
    mean=[59.06, 47.64, 0.43],
    std=[36.83, 31.71, 0.86],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/eye_area/train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/eye_area/val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
    dict(type='Accuracy', topk=1),
    dict(
        type='BinaryLabelMetric',
        positive_label=0,
        items=['precision', 'recall', 'f1-score']),
    dict(type='ConfusionMatrix'),
    dict(
        type='AUC',
        average='macro',
        plot_roc=True,
        roc_save_path='work_dirs/resnet50-eye/roc_curve_val.png',
        positive_label=0,
        save_per_eval=True)
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = [
    dict(type='Accuracy', topk=1),
    dict(
        type='BinaryLabelMetric',
        positive_label=0,
        items=['precision', 'recall', 'f1-score']),
    dict(type='ConfusionMatrix'),
    dict(type='AUC',
         average='macro',
         plot_roc=True,
         roc_save_path='work_dirs/resnet50-eye/roc_curve_test.png',
         positive_label=0,
         save_per_eval=True)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        bias_decay_mult=0.,
        norm_decay_mult=0.),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=5,
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-7, by_epoch=True, begin=5)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

custom_hooks = [dict(type='ROCCurveHook')]

# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
