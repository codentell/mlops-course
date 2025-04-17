import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd


def load_and_introduce_missing(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Introduce missing values into 10% of the data
    rng = np.random.default_rng(seed=42)
    mask = rng.random(df.shape) < 0.1
    df = df.mask(mask)

    return df

def impute_data(df):
    imputer = SimpleImputer(strategy='mean')
    X = df.drop(columns='target')
    y = df['target']
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X_imputed, y