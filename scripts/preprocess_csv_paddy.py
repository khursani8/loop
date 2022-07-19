import pandas as pd
import numpy as np
from pathlib import Path
import os
from  sklearn.model_selection  import StratifiedKFold,StratifiedGroupKFold

files = os.listdir()

if "input" in files:
    input_dir = Path('input/')
else:
    input_dir = Path('../input/')

train = pd.read_csv(input_dir/"train.csv")
test = pd.read_csv(input_dir/"sample_submission.csv")
group = False
n_folds = 5

if group:
    skf = StratifiedGroupKFold(n_splits=n_folds, random_state=1111, shuffle=True)
    splits = skf.split(np.arange(len(train)), y=train.label.values, groups=train.variety.values)
else:
    skf = StratifiedKFold(n_splits=n_folds, random_state=1111, shuffle=True)
    splits = skf.split(np.arange(len(train)), y=train.label.values)
train["fold"] = -1

for fold, (train_set, val_set) in enumerate(splits):
    train.loc[train.index[val_set], "fold"] = fold

# need to follow what ImageDataset class expected
train["path"] = train.apply(lambda x:f'{input_dir}/train_images/{x["label"]}/{x["image_id"]}',axis=1)
test["path"] = test.apply(lambda x:f'{input_dir}/test_images/{x["image_id"]}',axis=1)

labels = sorted(train.label.unique())
lbl2id = {i:idx for idx,i in enumerate(labels)}
with open(input_dir / "labels.txt","w+") as o:
    o.write(str(labels))
with open(input_dir / "lbl2id.txt","w+") as o:
    o.write(str(lbl2id))
train["target"] = train.label.apply(lambda x:lbl2id[x])

test['fold'] = 0
test['target'] = 0
test['sorting_date'] = 0

train.to_csv(input_dir / 'train_with_fold.csv', index=False)
test.to_csv(input_dir / 'test_with_fold.csv', index=False)