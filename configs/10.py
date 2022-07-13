from paddy_base import *

target_size = (480,480) # None=Original size.
normalize = False
batch_size = 64*num_gpu
num_workers = 8

ema = dict(
    beta=0.995
)

# model config
# model
model = dict(
    name = 'TimmModel',
    arch = 'efficientnetv2_rw_t',
    num_classes = 10,
    params = dict(
    )
)

sam_optimizer = False
# optimizer
optim = dict(
    name = 'AdamW',
    lr = 0.001*num_gpu,
    weight_decay = 1e-4
)

# loss
loss = dict(
    name = 'LabelSmoothingCrossEntropy',
    params = dict(
    ),
)

# metric
metric = dict(
    name = 'accuracy',
    params = dict(
    ),
)

# scheduler
scheduler = dict(
    name = 'CosineAnnealingLR',
    params = dict(
        T_max=epochs*10,
        eta_min=0,
        last_epoch=-1,
    )
)

from augmentation import *

# train.
train = dict(
    is_valid = False,
    data_path = input_dir / f'train_with_fold.csv',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    loader=dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [
        horizontalflip,
        shiftscalerotate,
        advblur,
        blur,
        randombrightnesscontrast,
        cutout,
        normalize,
        totensor
        ],
)


# valid.
valid = dict(
    is_valid = True,
    data_path = input_dir / f'train_with_fold.csv',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [normalize,totensor],
)


# test.
test = dict(
    is_valid = True,
    data_path = input_dir / 'test_with_fold.csv',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [normalize,totensor],
)