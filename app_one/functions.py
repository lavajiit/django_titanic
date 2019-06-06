from sklearn.externals import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


def preprocess_train(df):
    df['Embarked'] = df['Embarked'].fillna('C')
    df = pd.get_dummies(df, columns=["Pclass", "Embarked"])
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    mean = df["Age"].mean()
    std = df["Age"].std()
    is_null = df["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age_slice = df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    df["Age"] = age_slice
    df.loc[df['Age'] <= 13, 'Age'] = 0
    df.loc[(df['Age'] > 13) & (df['Age'] <= 30), 'Age'] = 1
    df.loc[(df['Age'] > 30) & (df['Age'] <= 43), 'Age'] = 2
    df.loc[df['Age'] > 43, 'Age'] = 3
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    df.loc[df['Fare'] <= 7, 'Fare'] = 0
    df.loc[(df['Fare'] > 7) & (df['Fare'] <= 14), 'Fare'] = 1
    df.loc[(df['Fare'] > 14) & (df['Age'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Master'], 'Mr')
    df['Title'] = df['Title'].replace(['Ms', 'Mlle'], 'Miss')
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Mme', 'Dr', 'Rev', 'Major', 'Col', 'Don', 'Dona', 'Jonkheer', 'Sir', 'Capt'], 'Rare')
    df = pd.get_dummies(df, columns=["Title"])
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin',
                  'SibSp', 'Parch', 'FamilySize'], 1)
    return df


def preprocess_test(df):
    # df['Embarked'] = df['Embarked'].fillna('C')
    df = pd.get_dummies(df, columns=["Pclass", "Embarked"])
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    # mean = df["Age"].mean()
    # std = df["Age"].std()
    # is_null = df["Age"].isnull().sum()
    # rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # age_slice = df["Age"].copy()
    # age_slice[np.isnan(age_slice)] = rand_age
    # df["Age"] = age_slice
    df.loc[df['Age'] <= 13, 'Age'] = 0
    df.loc[(df['Age'] > 13) & (df['Age'] <= 30), 'Age'] = 1
    df.loc[(df['Age'] > 30) & (df['Age'] <= 43), 'Age'] = 2
    df.loc[df['Age'] > 43, 'Age'] = 3
    # df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    df.loc[df['Fare'] <= 7, 'Fare'] = 0
    df.loc[(df['Fare'] > 7) & (df['Fare'] <= 14), 'Fare'] = 1
    df.loc[(df['Fare'] > 14) & (df['Age'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Master'], 'Mr')
    df['Title'] = df['Title'].replace(['Ms', 'Mlle'], 'Miss')
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Mme', 'Dr', 'Rev', 'Major', 'Col', 'Don', 'Dona', 'Jonkheer', 'Sir', 'Capt'], 'Rare')
    df = pd.get_dummies(df, columns=["Title"])
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin',
                  'SibSp', 'Parch', 'FamilySize'], 1)
    return df


def save_model(model):
    joblib.dump(model, 'model.pkl')
    print("Model dumped!")


def load_model():
    model = joblib.load('model.pkl')
    print("Model Loaded!")
    return model


def save_columns(model_columns):
    joblib.dump(list(model_columns), 'model_columns.pkl')
    print("Columns dumped!")


def load_columns():
    columns = joblib.load('model_columns.pkl')
    print("Columns Loaded!")
    return columns
