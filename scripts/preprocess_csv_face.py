import pandas as pd
import numpy as np
from pathlib import Path
import os
from  sklearn.model_selection  import KFold, StratifiedKFold,StratifiedGroupKFold,RepeatedStratifiedKFold

def mapper(age,celebrity):
    hobby = None
    if celebrity == "Actress":
        if age >= 15 and age<20:
            hobby = "Coloring"
        elif age >= 20 and age<25:
            hobby = "Acting"
        elif age >= 25 and age<30:
            hobby = "Gaming"
        elif age >= 30 and age<35:
            hobby = "Fashion"
        elif age >= 35 and age<40:
            hobby = "Embroidery"
        elif age >= 40 and age<45:
            hobby = "Programming"
        elif age >= 45 and age<50:
            hobby = "Pottery"
        elif age >= 50 and age<55:
            hobby = "Calligraphy"
        elif age >= 55 and age<60:
            hobby = "Painting"
        elif age >= 60 and age<65:
            hobby = "Sports"
        elif age >= 65 and age<70:
            hobby = "Archery"
        elif age >= 70 and age<75:
            hobby = "Reading"
        elif age >= 75 and age<80:
            hobby = "Skating"
        elif age >= 80 and age<85:
            hobby = "Electronics"
        elif age >= 85 and age<90:
            hobby = "Cycling"
        elif age >= 90 and age<95:
            hobby = "scrapbook"
        elif age >= 95 and age<100:
            hobby = "Puzzles"
        elif age >= 100 and age<105:
            hobby = "Cooking"
    else:
        if age >= 15 and age<20:
            hobby = "Vacation"
        elif age >= 20 and age<25:
            hobby = "Language"
        elif age >= 25 and age<30:
            hobby = "Origami"
        elif age >= 30 and age<35:
            hobby = "Writing"
        elif age >= 35 and age<40:
            hobby = "Sketching"
        elif age >= 40 and age<45:
            hobby = "Drama"
        elif age >= 45 and age<50:
            hobby = "Dance"
        elif age >= 50 and age<55:
            hobby = "television"
        elif age >= 55 and age<60:
            hobby = "Sewing"
        elif age >= 60 and age<65:
            hobby = "Pet"
        elif age >= 65 and age<70:
            hobby = "Sculpting"
        elif age >= 70 and age<75:
            hobby = "Sudoku"
        elif age >= 75 and age<80:
            hobby = "Magic"
        elif age >= 80 and age<85:
            hobby = "Yoga"
        elif age >= 85 and age<90:
            hobby = "Singing"
        elif age >= 90 and age<95:
            hobby = "Coffee"
        elif age >= 95 and age<100:
            hobby = "Drawing"
        elif age >= 100 and age<105:
            hobby = "Cryptography"
    return hobby

files = os.listdir()

if "input" in files:
    input_dir = Path('input/face/')
else:
    input_dir = Path('../input/face/')

train = pd.read_csv(input_dir/"train_post.csv")
test = pd.read_csv(input_dir/"sample_submission.csv")
group = False
strat = True
oversampling = False
fix_missing_label = False
n_folds = 4

classes = train.label.unique()
print(classes)
print(f"Total classes:{len(classes)}")

res = None
sample_to = train.label.value_counts().max()
print(train.label.value_counts())

if oversampling:
    for grp in train.groupby('label'):
        n = grp[1].shape[0]
        additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)
        rows = pd.concat((grp[1], additional_rows))

        if res is None: res = rows
        else: res = pd.concat((res, rows))
        train = res

print(train.label.value_counts())

if group:
    skf = StratifiedGroupKFold(n_splits=n_folds, random_state=1111, shuffle=True)
    splits = skf.split(np.arange(len(train)), y=train.label.values, groups=train.age.values)
else:
    if strat:
        skf = StratifiedKFold(n_splits=n_folds, random_state=1111, shuffle=True)
        splits = skf.split(np.arange(len(train)), y=train.label.values)
    else:
        skf = KFold(n_splits=n_folds,random_state=1111,shuffle=True)
        splits = skf.split(np.arange(len(train)),y=train.label.values)
train["fold"] = -1

train["label"] = train.apply(lambda x:mapper(x["age"],x["celebrity"]),axis=1)

for fold, (train_set, val_set) in enumerate(splits):
    train.loc[train.index[val_set], "fold"] = fold

for i in range(n_folds):
    valid_class = train[train.fold==i].label.unique()
    fold_cls_len = len(valid_class)
    trn_class = train[train.fold!=i].label.unique()
    trn_cls_len = len(trn_class)
    if fold_cls_len != trn_cls_len:
        missing_class = set(classes) ^ set(trn_class)
        print(i,"error",fold_cls_len,trn_cls_len,len(classes),missing_class)
        if missing_class and fix_missing_label:
            for cls in missing_class:
                rows = train[train.label == cls]
                rows.loc[rows.index,"fold"] = i - 1
                print(train.iloc[-1])
                train = pd.concat([train,rows],ignore_index=True,axis=0)
                print(train.iloc[-1])
        # if len(classes) != trn_cls_len:


# need to follow what ImageDataset class expected
train["path"] = train.apply(lambda x:f'{input_dir}/images/images/{x["id"]}.jpg',axis=1)
test["path"] = test.apply(lambda x:f'{input_dir}/images/images/{x["id"]}.jpg',axis=1)

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