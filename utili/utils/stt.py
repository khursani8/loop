from random import sample
import re
import soundfile as sf
import librosa
import torch
from pathlib import Path
import pandas as pd
import os
from pqdm.threads import pqdm

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_files(path, extensions=None, recurse=True, folders=[], followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return res

def process_text(a):
    a = re.sub(r'\([^)]*\)', '', a)
    a = re.sub(r'<[^<]+>', '', a)
    a = re.sub(r'{[^{]+}', '', a)
    a = re.sub(r'[\[\]]+','',a)
    to_replace = ".,?!-'():;+"
    for symbol in to_replace:
        a = a.replace(symbol,"")
    text = a.replace('"',"").replace("\\"," ").replace("/"," ").replace("*","").replace("&","dan").replace("$","dollar").lower()

    # if not isinstance(text,str):
    #     set_trace()
    text = text.strip()
    text = filter(lambda x:x!="",text.split(" "))
    text = " ".join(text)
    if text == "":
        text = " "
    return text

def collect_data(i, skip_empty=False):
    try:
        dur = librosa.get_duration(filename=str(i))
        if dur > 12 or dur < 1:
            return
        text = process_text(i.with_suffix(".txt").read_text())
        if text == "":
            if skip_empty:
                return
            else:
                return [i, "", 1, dur]
        return [i, text, 1, dur]
    except Exception as e:
        print(f"error {e}")
        return  # [i,"",1]


def get_item(path, i=0, aug=None):
    # f = preprocess_wav(path)
    f, samplerate = sf.read(str(path))
    # if aug:
    #     try:
    #         f = aug(f)
    #     except:
    #         pass
    fl = torch.tensor(f.shape[0])
    return f, fl, i


def get_df(source, recalculate=False):
    source = Path(source)
    cache = Path(f"/content/{source.name}.csv")
    if cache.is_file():
        print("use cache,", cache)
        df = pd.read_csv(cache)
        if not recalculate:
            return df
    else:
        df = []
    w = get_files(source, [".wav"], recurse=True)
    if len(w) == 0:
        return w
    if len(df) < len(w) - 100 or "test" in str(source):
        if len(df):
            print("recalculate", source, len(df), len(w))
        data = pqdm(w, collect_data, n_jobs=4)
        print(f"unfiltered:{len(data)}")
        data = list(filter(lambda x: x, data))  # remove the filter out data
        print(f"filtered  :{len(data)}")
        df = pd.DataFrame(data)
        df.columns = ["path", "text", "prob", "dur"]
        df["prob"] = pd.to_numeric(df["prob"])
        df.to_csv(f"/content/{source.name}.csv", index=False)
    df["path"] = df["path"].apply(lambda x: str(x))
    return df




class SafeDataset(torch.utils.data.Dataset):  # steal from nunchucks
    """A wrapper around a torch.utils.data.Dataset that allows dropping
    samples dynamically.
    """

    def __init__(self, dataset, eager_eval=False):
        """Creates a `SafeDataset` wrapper around `dataset`."""
        self.dataset = dataset
        self.eager_eval = eager_eval
        # These will contain indices over the original dataset. The indices of
        # the safe samples will go into _safe_indices and similarly for unsafe
        # samples.
        self._safe_indices = []
        self._unsafe_indices = []

        # If eager_eval is True, we can simply go ahead and build the index
        # by attempting to access every sample in self.dataset.
        if self.eager_eval is True:
            self._build_index()

    def _safe_get_item(self, idx):
        """Returns None instead of throwing an error when dealing with an
        unsafe sample, and also builds an index of safe and unsafe samples as
        and when they get accessed.
        """
        try:
            # differentiates IndexError occuring here from one occuring during
            # sample loading
            invalid_idx = False
            if idx >= len(self.dataset):
                invalid_idx = True
                raise IndexError
            sample = self.dataset[idx]
            if idx not in self._safe_indices:
                self._safe_indices.append(idx)
            return sample
        except Exception as e:
            if isinstance(e, IndexError):
                if invalid_idx:
                    raise
            if idx not in self._unsafe_indices:
                self._unsafe_indices.append(idx)
            return None

    def _build_index(self):
        for idx in range(len(self.dataset)):
            # The returned sample is deliberately discarded because
            # self._safe_get_item(idx) is called only to classify every index
            # into either safe_samples_indices or _unsafe_samples_indices.
            _ = self._safe_get_item(idx)

    def _reset_index(self):
        """Resets the safe and unsafe samples indices."""
        self._safe_indices = self._unsafe_indices = []

    @property
    def is_index_built(self):
        """Returns True if all indices of the original dataset have been
        classified into safe_samples_indices or _unsafe_samples_indices.
        """
        return len(self.dataset) == len(self._safe_indices) + len(self._unsafe_indices)

    @property
    def num_samples_examined(self):
        return len(self._safe_indices) + len(self._unsafe_indices)

    def __len__(self):
        """Returns the length of the original dataset.
        NOTE: This is different from the number of actually valid samples.
        """
        return len(self.dataset)

    def __iter__(self):
        return (
            self._safe_get_item(i)
            for i in range(len(self))
            if self._safe_get_item(i) is not None
        )

    def __getitem__(self, idx):
        """Behaves like the standard __getitem__ for Dataset when the index
        has been built.
        """
        argmaxed = 0
        while idx < len(self.dataset):
            sample = self._safe_get_item(idx)
            if sample is not None:
                return sample
            if not argmaxed:
                # idx = self.dataset.data["prob"].argmax()
                idxs = (
                    self.dataset.data["prob"]
                    .nlargest(int(len(self.dataset) * 0.01))
                    .index.tolist()
                )
                idx = random.choice(idxs)
                argmaxed += 1
            else:
                idx += 1
                if len(self.dataset) - 1 < idx:
                    idx -= 2
                argmaxed += 1
            if argmaxed > len(self.dataset) * 0.05:  # sampling for 5% of the data
                return sample
                raise IndexError

        raise IndexError

    def __getattr__(self, key):
        """Delegates to original dataset object if an attribute is not
        found in this class.
        """
        return getattr(self.dataset, key)

from pqdm.threads import pqdm

def decode(out,length,labels):
    length = length.int()
    out = out[:length].cpu().numpy()
    out2 = ["_"]+list(out)
    collapsed = []
    for idx,i in enumerate(out):
        if i!=out2[idx] and i!=len(labels)-1:
            collapsed.append(i)
    return "".join([labels[i] for i in collapsed]).replace("_","")

def decodes(output,ln,labels):
    if breakpoint:
        n_jobs = 1
    else:
        n_jobs = 4
    return pqdm(zip(output,ln,[labels]*len(output)),decode,n_jobs=n_jobs,argument_type='args',disable=True)

def flatten(xss):
    return [x for xs in xss for x in xs]