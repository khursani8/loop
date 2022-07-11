# References:
# https://github.com/fastai/fastai/blob/master/fastai/metrics.py

import sklearn.metrics as skm
import torch
def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    targ = torch.as_tensor(targ)
    if axis is not None:
        pred = inp.argmax(axis)
    else:
        pred = torch.as_tensor(inp)
    return (pred == targ).float().mean()

def error_rate(inp, targ, axis=-1):
    "1 - `accuracy`"
    return 1 - accuracy(inp, targ, axis=axis)

def top_k_accuracy(inp, targ, k=5, axis=-1):
    "Computes the Top-k accuracy (`targ` is in the top `k` predictions of `inp`)"
    inp = inp.topk(k=k, dim=axis)[1]
    targ = targ.unsqueeze(dim=axis).expand_as(inp)
    return (inp == targ).sum(dim=-1).float().mean()

def f1_score(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "F1 score for single-label classification problems"
    return skm.f1_score(axis=axis,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)

def fbeta_score(beta, axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "FBeta score with `beta` for single-label classification problems"
    return skm.fbeta_score(axis=axis,
                beta=beta, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)