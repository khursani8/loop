import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
from utili import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()

args = get_args()
cfg = utils.Config.fromfile(args.config)

version = args.version
folds = list(range(5))

sub = pd.read_csv("input/sample_submission.csv")
pred = np.zeros((len(sub), cfg.model.num_classes))

train = pd.read_csv(cfg.train.data_path)
target = train.target.values
oof = pd.DataFrame()
oof_cls = pd.DataFrame()

for fold in folds:
    df = pd.read_csv(f'output/{version}/{fold}/sub_{version}_raw.csv')
    #if df.target.isnull().sum():
    #    print('null detected!!')
    #    df.target = df.target.fillna(0)
    pred += df.values
    oof_raw_name = [x for x in os.listdir(f'output/{version}/{fold}/') if ('oof' in x)&('raw' in x)]
    oof_cls_name = [x for x in os.listdir(f'output/{version}/{fold}/') if ('oof' in x)&('raw' not in x)]

    print(oof_raw_name)
    print(oof_cls_name)

    _oof = pd.read_csv(f'output/{version}/{fold}/{oof_raw_name[0]}')
    oof = pd.concat([oof, _oof], axis=0)

    _oof = pd.read_csv(f'output/{version}/{fold}/{oof_cls_name[0]}')
    oof_cls = pd.concat([oof_cls, _oof], axis=0)

def get_pred(pred):
    return np.array([(x*range(cfg.model.num_classes)).mean() for x in pred])
cols = [f"conf_{i}" for i in range(10)]
oof.columns = cols
print(oof)
_oof = oof.copy()
_oof.conf_0 *= 0
_oof.conf_1 *= 1
_oof.conf_2 *= 2
_oof.conf_3 *= 3
_oof.conf_4 *= 4
_oof.conf_5 *= 5
_oof.conf_6 *= 6
_oof.conf_7 *= 7
_oof.conf_8 *= 8
_oof.conf_9 *= 9
_oof['pred'] = np.sum(_oof.values, axis=1)

oof_pred = _oof.pred.values
oof_target = oof_cls.target

print(oof_pred.shape, oof_target.shape)

assert len(oof)==len(train)
score = np.sqrt(mean_squared_error(oof_target, oof_pred))
print(score)

oof_cls.pred = oof_pred
oof_cls = pd.concat([oof_cls, oof], axis=1)

pred /= len(folds)

sub["target"] = (pred * range(cfg.model.num_classes)).sum(axis=1)
print(sub.target)

sub[cols] = pred['target']
sub[['target']+cols].to_csv(f'output/sub_{version}.csv', index=False)
oof_cls.to_csv(f'output/oof_{version}_{score:.4f}.csv', index=False)
