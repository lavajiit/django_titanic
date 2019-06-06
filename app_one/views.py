from django.shortcuts import render
from .functions import preprocess_train, preprocess_test, load_model, load_columns
import pandas as pd
import numpy as np
from sklearn.externals import joblib


def home(request):
    return render(request, 'app_one/home.html')


def result(request):
    if request.method == 'POST':
        PassengerId = request.POST.get('PassengerId')
        Pclass = request.POST.get('Pclass')
        Name = request.POST.get('Name')
        Sex = request.POST.get('Sex')
        Age = request.POST.get('Age')
        SibSp = request.POST.get('SibSp')
        Parch = request.POST.get('Parch')
        Ticket = request.POST.get('Ticket')
        Fare = request.POST.get('Fare')
        Cabin = request.POST.get('Cabin')
        Embarked = request.POST.get('Embarked')

        context = {
            'PassengerId': PassengerId,
            'Pclass': Pclass,
            'Name': Name,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Ticket': Ticket,
            'Fare': Fare,
            'Cabin': Cabin,
            'Embarked': Embarked,
        }
        # Create pandas dataframe from the context dictionary
        mini_df = pd.DataFrame(context, index=[0])
        mini_df['Age'] = mini_df['Age'].astype(np.float)
        mini_df['Fare'] = mini_df['Fare'].astype(np.float)
        mini_df['SibSp'] = mini_df['SibSp'].astype(np.float)
        mini_df['Parch'] = mini_df['Fare'].astype(np.float)

        # preprocess the data
        processed = preprocess_test(mini_df)

        # Load columns----------------------------------------------------
        model_columns = load_columns()
        print(model_columns)

        # As columns in mini_df and model_columns are different, let's make them equal
        processed_cols = processed.reindex(
            columns=model_columns, fill_value=0)
        print(list(processed_cols.columns))

        # Load model------------------------------------------------------
        model = load_model()
        # Predict
        pred_y = model.predict(processed_cols)

    return render(request, 'app_one/result.html', {'pred': pred_y[0]})
