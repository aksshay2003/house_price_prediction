import pandas as pd
from prefect import task
from src.ingest_data import DataIngestorFactory

@task
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    file_extension = ".zip"  # Hardcoded since we're using ZIP files

    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)

    return df
