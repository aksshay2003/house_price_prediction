from prefect import flow, task
import click
from rich import print
import mlflow
import os
import subprocess
import time
import signal
import joblib
from steps.model_evaluator_step import model_evaluator_step
model_name="model_pickle"
model_path = f'./models/{model_name}_production.pkl'
# MLFLOW_MODEL_DIR = "mlruns_model"
MLFLOW_PORT = 5000


@task
def load_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


@task
def log_model_to_mlflow(model):
    with mlflow.start_run(run_name="manual_deploy_run"):
        # eval_metrics,mse=model_evaluator_step
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name=model_name)
        # mlflow.sklearn.log_metrics({
        #     'metrics':
        # })
        print("[green]Model logged to MLflow.[/green]")


@task
def serve_model():
    print("[blue]Starting MLflow prediction server...[/blue]")
    # Assumes mlflow tracking URI is local filesystem or DB
    process = subprocess.Popen([
        "mlflow", "models", "serve",
        "-m", f"models:/{model_name}/1",
        "-p", str(MLFLOW_PORT),
        "--no-conda"
    ])
    print(f"[green]MLflow server started at http://127.0.0.1:{MLFLOW_PORT}[/green]")
    with open("mlflow_server.pid", "w") as f:
        f.write(str(process.pid))


@task
def stop_model_server():
    if os.path.exists("mlflow_server.pid"):
        with open("mlflow_server.pid", "r") as f:
            pid = int(f.read())
            print(f"[yellow]Stopping MLflow server with PID {pid}...[/yellow]")
            os.kill(pid, signal.SIGTERM)
            os.remove("mlflow_server.pid")
            print("[green]MLflow server stopped.[/green]")
    else:
        print("[red]No server PID file found.[/red]")


@flow
def run_main_prefect(stop_service: bool = False):
    if stop_service:
        stop_model_server()
        return

    model = load_model()
    # log_model_to_mlflow(model)
    serve_model()


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def cli(stop_service: bool):
    run_main_prefect(stop_service=stop_service)


if __name__ == "__main__":
    cli()
