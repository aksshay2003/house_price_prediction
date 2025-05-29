import logging
import os
import joblib
from typing import Optional

import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from prefect import task

@task
def model_building_step(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Pipeline:
    """
    Builds, trains a Linear Regression model pipeline and saves it as a pickle file.

    Parameters:
    X_train (pd.DataFrame): Training data features.
    y_train (pd.Series): Training data labels/target.
    model_name (str): Name for the saved model pickle file.

    Returns:
    Pipeline: The trained sklearn pipeline.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Building and training the Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Ensure the models directory exists
        os.makedirs('./models', exist_ok=True)

        # Save the pipeline as a pickle file
        model_name="model_pickle"
        model_path = f'./models/{model_name}_production.pkl'
        joblib.dump(pipeline, model_path)
        logging.info(f"Model pipeline saved to {model_path}")

    except Exception as e:
        logging.error(f"Error during model training or saving: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline
