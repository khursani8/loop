from stt_base import *
from utili.utils.collate import collate_1d

pref = "lala"

batch_size = 64*num_gpu
num_workers = 0
# epochs=20

ema = dict(
    beta=0.9
)

# model config
# model
encoder = dict(
        name = 'SqueezeformerEncoder',
        params = dict(
            feat_in = 80,
            n_layers = 16,
            d_model = 144,
            feat_out = -1,
            subsampling = 'dw_striding',
            subsampling_factor = 4,
            subsampling_conv_channels = -1,
            ff_expansion_factor = 4,
            self_attention_model = 'rel_pos',
            n_heads = 4,
            att_context_size = [-1, -1],
            xscaling = True,
            untie_biases = True,
            pos_emb_max_len = 5000,
            conv_kernel_size = 31,
            conv_norm_type = 'batch_norm',
            dropout = 0.1,
            dropout_emb = 0.,
            dropout_att = 0.1,
            adaptive_scale = True,
            time_reduce_idx = 7,
            time_recovery_idx = None,
        )
    )

model = dict(
    name = 'SpeechModel',
    frontend = dict(
        name = "MelSpec",
        params = dict(),
    ),
    encoder=encoder,
    decoder = dict(
        name = 'Linear',
        params = dict(
            in_features = encoder["params"]["d_model"],
            out_features = len(labels)
        )
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
    name = 'CTCLoss',
    params = dict(
    ),
)

# metric
metric = dict(
    name = 'cer',
    params = dict(
    ),
)

# scheduler
# scheduler = dict(
#     name = 'CosineAnnealingLR',
#     params = dict(
#         # T_max=epochs*10,
#         T_max=200,
#         eta_min=0,
#         last_epoch=-1,
#     )
# )

scheduler = dict(
    name = 'OneCycleLR',
    params = dict(
        max_lr=optim["lr"],
        epochs=epochs
    )
)

import torch
def collate_fn(batch,pad_idx=0):
    audio = collate_1d([s['audio'] for s in batch], pad_idx)
    audio_len = torch.LongTensor([s['audio'].numel() for s in batch])
    target = collate_1d([s['target'] for s in batch], pad_idx)
    target_len = torch.LongTensor([s['target'].numel() for s in batch])
    return {
        "audio":audio,
        "audio_len":audio_len,
        "target":target,
        "target_len":target_len
    }

# train.
train = dict(
    lbl2idx=lbl2idx,
    is_valid = False,
    source = [
        "/mnt/1CFE87062D2332ED/malayaspeech/malay",
        "/mnt/1CFE87062D2332ED/malayaspeech/singlish",
    ],
    dataset_name = dataset_name,
    loader=dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        ),
    # collate_fn = ,
)


# valid.
valid = dict(
    lbl2idx=lbl2idx,
    source = [
        "/content/test"
    ],
    is_valid = True,
    dataset_name = dataset_name,
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        ),
)


# test.
test = dict(
    is_valid = True,
    data_path = input_dir / 'test_with_fold.csv',
    dataset_name = dataset_name,
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        ),
)