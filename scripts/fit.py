# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib

# обучение модели
def fit_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
        
    data = pd.read_csv('data/initial_data.csv')
	# загрузите результат предыдущего шага: inital_data.csv

# обучение модели
    cat_features = data.select_dtypes(include='object')
    # potential_binary_features = cat_features.nunique() == 2

    # binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    # other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(['float'])

    preprocessor = ColumnTransformer(
        [
            ('cat', OneHotEncoder(drop=params['one_hot_drop']), cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    model = LogisticRegression(penalty=params['penalty'], C=params['C'])

    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    pipeline.fit(data, data[params['target_col']])
	# реализуйте основную логику шага с использованием гиперпараметров
    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)
	# сохраните обученную модель в models/fitted_model.pkl


if __name__ == '__main__':
	fit_model()