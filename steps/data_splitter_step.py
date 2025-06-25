from typing import Tuple
import pandas as pd
from prefect import task

from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy

@task
def data_splitter_step(df: pd.DataFrame, target_column: str):
    important_features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF']

    # Optional: filter important features that are present in df.columns
    existing_features = [feat for feat in important_features if feat in df.columns]

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in input data")

    df_reduced = df[existing_features + [target_column]]

    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df_reduced, target_column)

    return X_train, X_test, y_train, y_test
