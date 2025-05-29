from prefect import task
import json
import requests
import pandas as pd
import numpy as np

@task
def predictor(service_url: str, input_data: str) -> np.ndarray:
    """
    Call MLflow deployed model REST endpoint for prediction.

    Args:
        service_url (str): The MLflow model's deployed endpoint URL.
        input_data (str): JSON string with input data.

    Returns:
        np.ndarray: Model predictions.
    """

    # Load JSON data
    data = json.loads(input_data)
    data.pop("columns", None)
    data.pop("index", None)

    # Expected columns
    expected_columns = [
        "Order", "PID", "MS SubClass", "Lot Frontage", "Lot Area", "Overall Qual",
        "Overall Cond", "Year Built", "Year Remod/Add", "Mas Vnr Area", "BsmtFin SF 1",
        "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF",
        "Low Qual Fin SF", "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath",
        "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces",
        "Garage Yr Blt", "Garage Cars", "Garage Area", "Wood Deck SF", "Open Porch SF",
        "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area", "Misc Val", "Mo Sold",
        "Yr Sold",
    ]

    # Format input
    df = pd.DataFrame(data["data"], columns=expected_columns)
    inputs = df.to_dict(orient="records")

    # MLflow expects input like: {"inputs": [...]}
    response = requests.post(
        url=service_url,
        headers={"Content-Type": "application/json"},
        json={"inputs": inputs}
    )

    if response.status_code != 200:
        raise RuntimeError(f"Prediction failed: {response.text}")

    predictions = response.json().get("predictions", [])
    return np.array(predictions)
