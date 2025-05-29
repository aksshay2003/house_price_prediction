from sklearn.pipeline import Pipeline
from prefect import task

@task
def model_loader(model_name: str) -> Pipeline:
    """
    Loads the current production model pipeline.

    Args:
        model_name: Name of the Model to load.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.
    """
    # Here Prefect doesn't have built-in model registry like ZenML,
    # so you need to implement your own model loading logic,
    # e.g., loading from a file system, MLflow, or cloud storage.

    # Example: loading a pickled sklearn pipeline from disk
    import joblib
    model_name="model_pickle"
    model_path = f'./models/{model_name}_production.pkl'
    model_pipeline: Pipeline = joblib.load(model_path)

    return model_pipeline
