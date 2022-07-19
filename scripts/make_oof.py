import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import f1_score, accuracy_score
import argparse
from utili import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()

pref = "face"
args = get_args()
cfg = utils.Config.fromfile(args.config)

version = args.version
folds = [0,1,2,3] # oof require all fold to be done

sub = pd.read_csv(f"input/{pref}/sample_submission.csv")
pred = np.zeros((len(sub), cfg.model.num_classes))

train = pd.read_csv(cfg.train.data_path)
target = train.target.values
oof = pd.DataFrame()
oof_cls = pd.DataFrame()

for fold in folds:
    df = pd.read_csv(f'output/{pref}/{version}/{fold}/sub_{version}_raw.csv')
    pred += df.values
    oof_raw_name = sorted([x for x in os.listdir(f'output/{pref}/{version}/{fold}/') if ('oof' in x)&('raw' in x)])
    oof_cls_name = sorted([x for x in os.listdir(f'output/{pref}/{version}/{fold}/') if ('oof' in x)&('raw' not in x)])

    print(oof_raw_name)
    print(oof_cls_name)

    _oof = pd.read_csv(f'output/{pref}/{version}/{fold}/{oof_raw_name[-1]}')
    oof = pd.concat([oof, _oof], axis=0)

    _oof = pd.read_csv(f'output/{pref}/{version}/{fold}/{oof_cls_name[-1]}')
    oof_cls = pd.concat([oof_cls, _oof], axis=0)

cols = [f"conf_{i}" for i in range(cfg.model.num_classes)]
oof.columns = cols

_oof = oof.copy()

_oof['pred'] = np.argmax(_oof.values, axis=1)

oof_pred = _oof.pred.values
oof_target = oof_cls.target

print(len(oof),len(train))
assert len(oof)==len(train)
score = f1_score(oof_target, oof_pred, average='micro')
print(score)

oof_cls.pred = oof_pred
oof_cls = pd.concat([oof_cls, oof], axis=1)

pred /= len(folds)

sub["target"] = np.argmax(pred, axis=1)

sub[cols] = pred
sub[['target']+cols].to_csv(f'output/sub_{version}.csv', index=False)
oof_cls.to_csv(f'output/oof_{version}_{score:.4f}.csv', index=False)
