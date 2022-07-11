import cv2
import torch
import pandas as pd
import numpy as np

class ImageDataset(torch.utils.data.Dataset):
    """
    This class expecting csv file with three columns
    1. path -> full path to the image
    2. target -> label for the image
    3. fold -> belong to which fold
    """
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        for i in ["path","target","fold"]:
            assert i in self.df.columns,f"{i} not in columns"

        self.df = self.df[self.df.fold.isin(folds)]

        # below is custom initializations.
        self.path = self.df.path.values
        self.target = self.df.target.values
        self.target_size = self.cfg.target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = self.target[idx]
        path = self.path[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255

        img = cv2.resize(img, self.target_size)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return {'image': img,
                'target': target}
