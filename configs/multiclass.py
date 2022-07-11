from pathlib import Path
from configs.base import *

dataset_name = "ImageDataset"
n_fold = 5
fold = 0
num_gpu = 1
gpu = "0"
n_epochs = 100

input_dir = Path(f'input')
output_dir = Path(f'output')

target_size = (224,224) # None=Original size.
normalize = False
batch_size = 64*num_gpu
num_workers = 8

# model config
# model
model = dict(
    name = 'TimmModel',
    arch = 'resnet18',
    pretrained_weight=None,
    num_classes = 10,
    params = dict(
        # pretrained_path='../dino/output/007/checkpoint.pth',
    )
)

# optimizer
optim = dict(
    name = 'AdamW',
    lr = 0.001*num_gpu,
    weight_decay = 0.01
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
        T_max=n_epochs,
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
        blur,
        randombrightnesscontrast,
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