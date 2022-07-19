from pathlib import Path
from string import ascii_lowercase
from configs.base import *

dataset_name = "AudioDataset"
n_fold = 1
num_gpu = 1
gpu = "0"
input_dir = Path(f'input')
output_dir = Path(f'output')
labels = ["_"," "] + list(ascii_lowercase)
# labels =  list(ascii_lowercase) + [" ","_"]
lbl2idx = {c: i for i, c in enumerate(labels)}