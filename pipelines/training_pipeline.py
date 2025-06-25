import os
import pickle
from prefect import flow
import mlflow
import dagshub

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step

@flow(name="ML Training Pipeline")
def ml_pipeline():
    """Prefect flow for end-to-end ML pipeline with MLflow tracking."""

    # Initialize DagsHub tracking and set MLflow tracking URI
    dagshub.init(repo_owner='aksshay2003', repo_name='my-first-repo', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/aksshay2003/my-first-repo.mlflow")

    mlflow.autolog()  # Enable auto logging
    mlflow.set_experiment("HousePricePrediction")

    with mlflow.start_run():
        # 1. Data Ingestion
        raw_data = data_ingestion_step(
            file_path="D:\\AI\\HousePrediction\\Data\\Dataset(zip).zip"
        )

        # 2. Missing Values
        filled_data = handle_missing_values_step(raw_data)

        # 3. Feature Engineering
        engineered_data = feature_engineering_step(
            filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
        )

        # 4. Outlier Removal
        clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

        # 5. Split
        X_train, X_test, y_train, y_test = data_splitter_step(
            clean_data, target_column="SalePrice"
        )

        # 6. Model Training
        model = model_building_step(X_train=X_train, y_train=y_train)

        # 7. Log model to MLflow
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # 8. Register model in MLflow registry (optional)
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="house_price_predictor"
        )

        # 9. Evaluation
        evaluation_metrics, mse = model_evaluator_step(
            trained_model=model, X_test=X_test, y_test=y_test
        )

        mlflow.log_metric("MSE", mse)
        for metric, value in evaluation_metrics.items():
            mlflow.log_metric(metric, value)

        # ✅ 10. Save model locally as pickle
        os.makedirs("models", exist_ok=True)
        with open("models/model_pickle_production.pkl", "wb") as f:
            pickle.dump(model, f)

        print("✅ Model also saved locally to models/model_pickle_production.pkl")

        return model
