from prefect import flow
import mlflow
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
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("HousePricePrediction")

    with mlflow.start_run():
        # Data Ingestion Step
        raw_data = data_ingestion_step(
            file_path="D:\\AI\\HousePrediction\\Data\\Dataset(zip).zip"
        )

        # Handling Missing Values
        filled_data = handle_missing_values_step(raw_data)

        # Feature Engineering
        engineered_data = feature_engineering_step(
            filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
        )

        # Outlier Detection
        clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

        # Data Splitting
        X_train, X_test, y_train, y_test = data_splitter_step(
            clean_data, target_column="SalePrice"
        )

        # Model Building
        model = model_building_step(X_train=X_train, y_train=y_train)

        # You can log the model here
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # if using `mlflow ui`

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            # registered_model_name="house_price_predictor"  we use this line if we dont want register_model_name()
        )
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="house_price_predictor"
        )


        # Model Evaluation
        evaluation_metrics, mse = model_evaluator_step(
            trained_model=model, X_test=X_test, y_test=y_test
        )

        mlflow.log_metric("MSE", mse)
        for metric, value in evaluation_metrics.items():
            mlflow.log_metric(metric, value)

        return model
