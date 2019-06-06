import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.random.seed(37)
df = pd.read_csv("/home/f/Data/Titanic_Data/train.csv")
print(df.columns)