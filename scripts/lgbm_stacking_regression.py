import numpy as np
import pandas as pd
from pathlib import Path
import os
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

Path('output/ensemble').mkdir(parents=True,exist_ok=True)

pref = 'paddy'
versions = ["01"]

sort_by = "path"
cols = [f"conf_{i}" for i in range(10)]

for i, version in enumerate(versions):
    oof_name = [x for x in os.listdir(f'output/') if f'oof_{version}' in x]
    print(oof_name)

    if i==0:
        oof = pd.read_csv(f'output/{oof_name[0]}').sort_values(sort_by)
        oof[f'pred_{i}'] = oof.pred.values
        oof[cols] = 0

        sub = pd.read_csv(f'output/sub_{version}.csv')
        sub[f'pred_{i}'] = sub.target.values
        sub[cols] = 0

    else:
        _tmp = pd.read_csv(f'output/{oof_name[0]}').sort_values(sort_by)
        oof[f'pred_{i}'] = _tmp.pred.values
        if version in ['081', '082']:
            oof[cols] += 0.5*_tmp[cols].values

        _tmp = pd.read_csv(f'output/sub_{version}.csv')
        sub[f'pred_{i}'] = _tmp.target.values
        if version in ['081', '082']:
            sub[cols] += 0.5*_tmp[cols].values

oof = oof.reset_index(drop=True)

print(oof.head())
print(sub.head())


params = {
    'objective': 'rmse',
    'metrics': 'rmse',
    'n_estimators': 10000,
    'boosting_type': 'gbdt',
    'num_leaves': 32,
    'max_depth': 2,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.3,
    'bagging_freq': 5,
    'reg_alpha': 0.5,
    'reg_lambda': 0,
}

stacking_oof = np.zeros(len(oof))
stacking_sub = np.zeros(len(sub))

features = [f'pred_{i}' for i in range(len(versions))]
features.extend(cols)
print(features)

n_fold = 5

for fold in range(n_fold):
    print(f'{fold=}')

    trn = oof[oof.fold!=fold]
    val = oof[oof.fold==fold]

    val_idx = val.index

    trn_x = trn[features]
    trn_y = trn['target']
    val_x = val[features]
    val_y = val['target']

    tst_x = sub[features]

    model = LGBMRegressor(**params)
    model.fit(trn_x, trn_y, eval_metric='rmse',
             eval_set=[(val_x, val_y)],
             verbose=100, early_stopping_rounds=200)

    val_pred = model.predict(val_x)
    stacking_sub += model.predict(tst_x)

    stacking_oof[val_idx] = val_pred


oof.pred = stacking_oof
oof.pred = np.clip(oof.pred.values, 0.0, float(10))
print(oof.pred)
score = np.sqrt(mean_squared_error(oof.target.values, oof.pred.values))
print(f'{score:.6f}')

stacking_sub = stacking_sub / float(n_fold)
stacking_sub = np.clip(stacking_sub, 0.0, float(10))

sub_name = '_'.join(versions)
pd.DataFrame(stacking_sub, columns=['target']).to_csv(f'output/ensemble/{pref}_stacking_{sub_name}.csv', index=False)
oof.to_csv(f'output/ensemble/{pref}_oof_stacking_{sub_name}.csv', index=False)
