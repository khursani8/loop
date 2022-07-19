from pathlib import Path
from configs.base import *

dataset_name = "ImageDataset"
n_fold = 5
num_gpu = 1
gpu = "0"
input_dir = Path(f'input/face')
output_dir = Path(f'output/face')
labels = ['Acting', 'Archery', 'Calligraphy', 'Coffee', 'Coloring', 'Cryptography', 'Cycling', 'Dance', 'Drama', 'Drawing', 'Electronics', 'Embroidery', 'Fashion', 'Gaming', 'Language', 'Magic', 'Origami', 'Painting', 'Pet', 'Pottery', 'Programming', 'Puzzles', 'Reading', 'Sculpting', 'Sewing', 'Singing', 'Skating', 'Sketching', 'Sports', 'Sudoku', 'Vacation', 'Writing', 'Yoga', 'scrapbook', 'television']
print(len(labels))
