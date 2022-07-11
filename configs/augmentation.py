
# augmentations.
horizontalflip = dict(
    name = 'HorizontalFlip',
    params = dict()
)

verticalflip = dict(
    name = 'VerticalFlip',
    params = dict()
)

shiftscalerotate = dict(
    name = 'ShiftScaleRotate',
    params = dict(
        shift_limit = 0.1,
        scale_limit = 0.1,
        rotate_limit = 15,
    ),
)

gaussnoise = dict(
    name = 'GaussNoise',
    params = dict(
        var_limit = 5./255.
        ),
)

blur = dict(
    name = 'Blur',
    params = dict(
        blur_limit = 3
    ),
)

# randommorph = dict(
#     name = 'RandomMorph',
#     params = dict(
#         size = target_size,
#         num_channels = 1,
#     ),
# )

randombrightnesscontrast = dict(
    name = 'RandomBrightnessContrast',
    params = dict(),
)

griddistortion = dict(
    name = 'GridDistortion',
    params = dict(),
)

elastictransform = dict(
    name = 'ElasticTransform',
    params = dict(
        sigma = 50,
        alpha = 1,
        alpha_affine = 10
    ),
)

cutout = dict(
    name = 'Cutout',
    params = dict(
        num_holes=1,
        max_h_size=int(256*0.3),
        max_w_size=int(256*0.3),
        fill_value=0,
        p=0.7
    ),
)

totensor = dict(
    name = 'ToTensorV2',
    params = dict(),
)

oneof = dict(
    name='OneOf',
    params = dict(),
)

normalize = dict(
    name = 'Normalize',
    params = dict(
        max_pixel_value=1.0
    ),
)