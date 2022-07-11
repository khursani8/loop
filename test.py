from data.image import ImageDataset
import pandas as pd

df = pd.read_csv("input/train.csv")
data = ImageDataset(df)
print(data[0])