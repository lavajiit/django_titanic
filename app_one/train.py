from sklearn.ensemble import RandomForestClassifier
from functions import preprocess_train, preprocess_test, save_model, save_columns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


def train_it():
    train = pd.read_csv("/home/f/Data/Titanic_Data/train.csv")

    X = train.drop('Survived', 1)
    y = train.Survived

    X = preprocess_train(X)

    model = RandomForestClassifier()

    model.fit(X, y)
    # pred_y = model.predict(X)
    return model, X.columns


save_model(train_it()[0])
save_columns(train_it()[1])
