
from pathlib import Path

import pandas as pd
import torch
from utili.utils import stt

class SmallProb(Exception):
    pass

class AudioDataset(torch.utils.data.Dataset):
    """Some Information about AudioDataset"""
    def __init__(self,cfg,folds=None,aug=None):
        super(AudioDataset, self).__init__()
        self.cfg = cfg
        self.aug = aug

        if isinstance(self.cfg.source, list):
            s = "_".join([Path(i).name for i in self.cfg.source])
            fn = f"/content/{s}.csv"
            if Path(fn).is_file():
                df = pd.read_csv(fn)
            else:
                d = list(filter(lambda x: len(x), [stt.get_df(i) for i in cfg.source]))
                df = pd.concat(d, ignore_index=True)
        else:
            df = stt.get_df(cfg.source)
        self.data = df#.iloc[:1000]
        m, s = divmod(self.data.dur.sum(), 60)
        h, m = divmod(m, 60)
        print(
            f"{len(self.data)} of audio files with total duration of {int(h)}hours {int(m)}minutes {int(s)}seconds"
        )

    def __getitem__(self, i):
        file = self.data.iloc[i]
        path = file.path
        text = file.text
        if len(text) < 1:
            raise SmallProb # skip easy data
        audio, audio_len, index = stt.get_item(path, text, aug=self.aug)
        t = []
        for j in text:
            try:
                t.append(self.cfg.lbl2idx[j])
            except Exception as e:
                pass
                # print('exception',e)
        return {
            "audio":torch.tensor(audio,dtype=torch.float32),
            "target":torch.tensor(t,dtype=torch.int),
        }

    def __len__(self):
        return len(self.data)