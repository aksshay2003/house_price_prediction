from prefect import task
import mlflow

@task
def prediction_service_loader(pipeline_name: str, step_name: str):
    """
    Load MLflow prediction service info matching pipeline and step names.

    Args:
        pipeline_name: Name of the pipeline that deployed the model.
        step_name: Name of the pipeline step that deployed the model.

    Returns:
        info about the deployed model service or raises error if none.
    """

    # List registered model versions or deployments (example approach)
    client = mlflow.tracking.MlflowClient()

    # Example: find model deployments (assuming you use MLflow Model Registry or serving)
    # Since MLflow itself doesn't have a direct 'find_model_server' method,
    # you would implement logic to find the model serving URI or deployment info
    # based on your naming conventions.

    # Pseudo-code: fetch deployments metadata from your deployment tracking
    # This depends heavily on your MLflow deployment setup.

    # For example, you might query model versions with certain tags:
    models = client.search_model_versions(f"name='{pipeline_name}'")

    # Filter by step_name tag or metadata (this depends on how you store it)
    deployed_services = [
        m for m in models if m.tags.get("step_name") == step_name
    ]

    if not deployed_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} pipeline is currently running."
        )

    # Return the deployment info of the first matching service
    return deployed_services[0]
